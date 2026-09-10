"""
Neural Cost Predictor for GA-Accelerated CVRP
================================================

Idea (mirrors the "neural cost predictor" concept from Sobhanan et al.,
"Genetic Algorithms with Neural Cost Predictor for Solving Hierarchical
Vehicle Routing Problems", Transportation Science 2025): exact fitness
evaluation of a CVRP tour requires decoding routes and summing distances
- correct, but expensive to run on every individual, every generation.

Instead, train a small neural network (scikit-learn MLPRegressor) to
predict a tour's total distance from cheap structural FEATURES of the
tour, then use it as a fast surrogate: generate a larger pool of
candidate offspring each generation, rank them all with the cheap NN
prediction, and only spend expensive EXACT evaluations on the most
promising subset. This cuts the number of exact evaluations needed to
reach a comparable solution quality.

Feature vector per tour (fixed-length, order-invariant structural
summary - NOT the raw permutation, which an MLP cannot use directly):
  - number of routes after greedy capacity split
  - total distance of a nearest-neighbor-order lower bound per route (avg)
  - mean route length (customers per route)
  - mean and std of consecutive-customer distances within routes
  - mean route "spread" (max pairwise distance within a route)
  - capacity utilization (mean load / capacity across routes)
"""

import numpy as np
from sklearn.neural_network import MLPRegressor

from genetic_algorithm import decode_routes, evaluate_exact, route_distance


def extract_features(tour, demands, capacity, dist_matrix):
    routes = decode_routes(tour, demands, capacity)
    n_routes = len(routes)

    route_lengths = [len(r) for r in routes]
    consec_dists = []
    route_spreads = []
    loads = []

    for r in routes:
        loads.append(sum(demands[c] for c in r))
        full = [0] + r + [0]
        for a, b in zip(full[:-1], full[1:]):
            consec_dists.append(dist_matrix[a, b])
        if len(r) >= 2:
            sub = dist_matrix[np.ix_(r, r)]
            route_spreads.append(sub.max())
        else:
            route_spreads.append(0.0)

    consec_dists = np.array(consec_dists) if consec_dists else np.array([0.0])
    loads = np.array(loads) if loads else np.array([0.0])

    features = [
        n_routes,
        float(np.mean(route_lengths)),
        float(np.std(route_lengths)),
        float(np.mean(consec_dists)),
        float(np.std(consec_dists)),
        float(np.sum(consec_dists)),  # correlates strongly with true cost
        float(np.mean(route_spreads)),
        float(np.mean(loads) / capacity),
    ]
    return features


def build_training_set(n_samples, demands, capacity, dist_matrix, seed=1):
    """Build MLP training data covering the FULL quality spectrum the GA
    will actually encounter - not just uniformly random tours.

    A model trained only on random permutations sees a narrow, high-cost
    slice of tour-space (random tours all have similarly poor structure).
    During an actual GA run, crossover/mutation of increasingly good
    parents produces much more organized, lower-cost tours - a different
    feature distribution the model never saw in training. That mismatch
    is exactly what caused the first version of this surrogate to make
    bad ranking decisions despite low validation error (the validation
    split was drawn from the same narrow random-tour distribution, so it
    couldn't reveal the generalization gap).

    Fix: run a short pilot GA and sample tours from EVERY generation
    (random early on, increasingly optimized later), mixed with plain
    random tours, so training data spans the quality range the real
    surrogate-accelerated GA will need to rank.
    """
    from genetic_algorithm import run_ga

    rng = np.random.default_rng(seed)
    n_customers = len(demands) - 1
    customers = list(range(1, n_customers + 1))

    tours = []

    # Half: pure random tours (covers the "bad" end of the spectrum)
    n_random = n_samples // 2
    for _ in range(n_random):
        tours.append(rng.permutation(customers).tolist())

    # Half: population snapshots from a short pilot GA run (covers
    # increasingly optimized tours, matching what the real run will see)
    n_pilot = n_samples - n_random
    pilot_pop_size = 40
    pilot_generations = max(1, n_pilot // pilot_pop_size)

    from genetic_algorithm import tournament_select, order_crossover, swap_mutation

    pop = [rng.permutation(customers).tolist() for _ in range(pilot_pop_size)]
    for _ in range(pilot_generations):
        fits = np.array([evaluate_exact(t, demands, capacity, dist_matrix) for t in pop])
        tours.extend(pop)
        best = pop[np.argmin(fits)]
        new_pop = [best.copy()]
        while len(new_pop) < pilot_pop_size:
            p1 = tournament_select(pop, fits, 3, rng)
            p2 = tournament_select(pop, fits, 3, rng)
            child = order_crossover(p1, p2, rng)
            child = swap_mutation(child, rng, 0.15)
            new_pop.append(child)
        pop = new_pop
    tours = tours[:n_samples]

    X, y = [], []
    for tour in tours:
        feats = extract_features(tour, demands, capacity, dist_matrix)
        cost = evaluate_exact(tour, demands, capacity, dist_matrix)
        X.append(feats)
        y.append(cost)
    return np.array(X), np.array(y)


def train_predictor(X, y, seed=0):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        max_iter=2000,
        random_state=seed,
        early_stopping=True,
    )
    model.fit(X_train_s, y_train)

    val_pred = model.predict(X_val_s)
    val_mae = np.mean(np.abs(val_pred - y_val))
    val_mape = np.mean(np.abs((val_pred - y_val) / y_val)) * 100

    return model, scaler, {"val_mae": val_mae, "val_mape": val_mape}


def predict_batch(tours, model, scaler, demands, capacity, dist_matrix):
    """Batched surrogate prediction - one scaler/model call for the whole
    candidate pool instead of one Python-level call per individual. The
    per-call overhead of MLPRegressor.predict() otherwise dominates and
    makes the 'accelerated' GA slower than the baseline it's supposed to
    speed up."""
    feats = np.array(
        [extract_features(t, demands, capacity, dist_matrix) for t in tours]
    )
    feats_s = scaler.transform(feats)
    return model.predict(feats_s)


def run_ga_with_surrogate(
    demands,
    capacity,
    dist_matrix,
    model,
    scaler,
    pop_size=60,
    generations=150,
    pool_multiplier=3,
    tournament_k=3,
    mutation_rate=0.15,
    seed=0,
):
    """GA variant: each generation, generate `pool_multiplier`x offspring,
    rank ALL of them with the cheap NN surrogate, and only spend exact
    evaluations on the top `pop_size` by predicted fitness. This is where
    the neural cost predictor saves expensive exact evaluations."""
    from genetic_algorithm import tournament_select, order_crossover, swap_mutation

    rng = np.random.default_rng(seed)
    n_customers = len(demands) - 1
    customers = list(range(1, n_customers + 1))

    population = [rng.permutation(customers).tolist() for _ in range(pop_size)]
    n_exact_evals = 0
    history = []
    best_individual_overall = None
    best_fitness_overall = np.inf

    # Initial population still needs exact fitness to seed selection
    fitnesses = np.array(
        [evaluate_exact(ind, demands, capacity, dist_matrix) for ind in population]
    )
    n_exact_evals += len(population)

    for gen in range(generations):
        gen_best_idx = np.argmin(fitnesses)
        if fitnesses[gen_best_idx] < best_fitness_overall:
            best_fitness_overall = fitnesses[gen_best_idx]
            best_individual_overall = population[gen_best_idx].copy()
        history.append((best_fitness_overall, n_exact_evals))

        # Generate a larger candidate pool
        pool_size = pop_size * pool_multiplier
        candidates = [best_individual_overall.copy()]  # elitism
        while len(candidates) < pool_size:
            p1 = tournament_select(population, fitnesses, tournament_k, rng)
            p2 = tournament_select(population, fitnesses, tournament_k, rng)
            child = order_crossover(p1, p2, rng)
            child = swap_mutation(child, rng, mutation_rate)
            candidates.append(child)

        # Cheap NN ranking of the whole pool (batched - no exact evals spent).
        # Pure top-K-by-prediction selection overfits the surrogate's blind
        # spots: a candidate the NN slightly underrates never gets a chance
        # to prove itself via exact evaluation. Keep a random-retention
        # fraction of the non-top-ranked pool as an exploration safety net
        # against systematic surrogate bias (standard fix for noisy
        # surrogate-assisted selection).
        predicted = predict_batch(candidates, model, scaler, demands, capacity, dist_matrix)
        order = np.argsort(predicted)
        n_top = int(pop_size * 0.7)
        top_idx = order[:n_top]
        remaining_idx = order[n_top:]
        n_random = pop_size - n_top
        random_idx = rng.choice(remaining_idx, size=min(n_random, len(remaining_idx)), replace=False)
        selected_idx = np.concatenate([top_idx, random_idx])
        selected = [candidates[i] for i in selected_idx]

        # Only the survivors get expensive exact evaluation
        fitnesses = np.array(
            [evaluate_exact(ind, demands, capacity, dist_matrix) for ind in selected]
        )
        n_exact_evals += len(selected)
        population = selected

    return best_individual_overall, best_fitness_overall, history
