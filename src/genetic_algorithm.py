"""
Genetic Algorithm for CVRP (Giant-Tour Representation)
=========================================================

Encoding: each individual is a permutation of customer indices (a "giant
tour"). A greedy capacity-respecting split decoder turns the permutation
into a set of depot-to-depot routes: walk the tour in order, start a new
route whenever adding the next customer would exceed vehicle capacity.

Fitness = total distance across all routes (lower is better).

Operators:
- Selection: tournament selection
- Crossover: Order Crossover (OX) - preserves permutation validity
- Mutation: swap mutation
- Elitism: best individual always survives to the next generation

This module exposes both `evaluate_exact` (the true, expensive fitness
using the full distance matrix) and a route-decoding function, so the
same decoder is reused by both the baseline GA and the neural-predictor
accelerated GA in `neural_predictor.py`.
"""

import numpy as np


def decode_routes(tour, demands, capacity):
    """Split a giant tour (list of customer indices, 1-based, depot excluded)
    into capacity-feasible routes via greedy splitting."""
    routes = []
    current_route = []
    current_load = 0
    for customer in tour:
        d = demands[customer]
        if current_load + d > capacity:
            routes.append(current_route)
            current_route = [customer]
            current_load = d
        else:
            current_route.append(customer)
            current_load += d
    if current_route:
        routes.append(current_route)
    return routes


def route_distance(route, dist_matrix):
    """Distance of a single route: depot(0) -> customers -> depot(0)."""
    if not route:
        return 0.0
    total = dist_matrix[0, route[0]]
    for a, b in zip(route[:-1], route[1:]):
        total += dist_matrix[a, b]
    total += dist_matrix[route[-1], 0]
    return total


def evaluate_exact(tour, demands, capacity, dist_matrix):
    """True fitness: total distance across all decoded routes."""
    routes = decode_routes(tour, demands, capacity)
    return sum(route_distance(r, dist_matrix) for r in routes)


def tournament_select(population, fitnesses, k=3, rng=None):
    idx = rng.choice(len(population), size=k, replace=False)
    best = idx[np.argmin(fitnesses[idx])]
    return population[best]


def order_crossover(parent1, parent2, rng):
    """OX: preserves a random slice from parent1, fills the rest in
    parent2's relative order."""
    n = len(parent1)
    a, b = sorted(rng.choice(n, size=2, replace=False))
    child = [None] * n
    child[a:b] = parent1[a:b]
    fill_values = [c for c in parent2 if c not in child[a:b]]
    pos = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill_values[pos]
            pos += 1
    return child


def swap_mutation(individual, rng, rate=0.1):
    individual = individual.copy()
    if rng.random() < rate:
        i, j = rng.choice(len(individual), size=2, replace=False)
        individual[i], individual[j] = individual[j], individual[i]
    return individual


def run_ga(
    demands,
    capacity,
    dist_matrix,
    pop_size=60,
    generations=150,
    tournament_k=3,
    mutation_rate=0.15,
    seed=0,
    fitness_fn=None,
):
    """Baseline GA using `fitness_fn` (defaults to exact evaluation) for
    EVERY individual, every generation. Returns best tour, best fitness,
    and a per-generation history of (best_fitness, n_exact_evals_so_far).
    """
    rng = np.random.default_rng(seed)
    n_customers = len(demands) - 1  # exclude depot
    customers = list(range(1, n_customers + 1))

    if fitness_fn is None:
        fitness_fn = lambda tour: evaluate_exact(tour, demands, capacity, dist_matrix)

    population = [rng.permutation(customers).tolist() for _ in range(pop_size)]
    n_exact_evals = 0
    history = []
    best_individual_overall = None
    best_fitness_overall = np.inf

    for gen in range(generations):
        fitnesses = np.array([fitness_fn(ind) for ind in population])
        n_exact_evals += len(population)

        gen_best_idx = np.argmin(fitnesses)
        if fitnesses[gen_best_idx] < best_fitness_overall:
            best_fitness_overall = fitnesses[gen_best_idx]
            best_individual_overall = population[gen_best_idx].copy()
        history.append((best_fitness_overall, n_exact_evals))

        new_population = [best_individual_overall.copy()]  # elitism
        while len(new_population) < pop_size:
            p1 = tournament_select(population, fitnesses, tournament_k, rng)
            p2 = tournament_select(population, fitnesses, tournament_k, rng)
            child = order_crossover(p1, p2, rng)
            child = swap_mutation(child, rng, mutation_rate)
            new_population.append(child)
        population = new_population

    return best_individual_overall, best_fitness_overall, history
