# Genetic Algorithm + Neural Cost Predictor for CVRP

A from-scratch implementation of a genetic algorithm for the Capacitated
Vehicle Routing Problem (CVRP), extended with a neural-network surrogate
cost predictor intended to accelerate fitness evaluation — inspired by the
general idea behind *"Genetic Algorithms with Neural Cost Predictor for
Solving Hierarchical Vehicle Routing Problems"* (Sobhanan, Park, Park,
Kwon — Transportation Science, 2025). This is an independent, original
implementation built to explore and understand that idea, not a
reproduction of the paper's code.

## Problem

CVRP: one depot, N customers each with a demand, an unlimited fleet of
vehicles with a fixed capacity. Minimize total distance across all
routes needed to serve every customer without exceeding vehicle
capacity on any route.

- **Instance**: 40 customers, synthetically generated with a fixed seed
  (`src/vrp_instance.py`), saved in standard TSPLIB/CVRPLIB `.vrp`
  format so a real benchmark instance can be substituted without
  changing any other code.
- **Encoding**: giant-tour permutation of customers, decoded into
  capacity-feasible routes via greedy splitting.

## Methodology

1. **Baseline GA** (`src/genetic_algorithm.py`) — standard genetic
   algorithm: tournament selection, order crossover (OX), swap mutation,
   elitism. Every individual gets the true (exact) fitness — total
   route distance — every generation.

2. **Neural cost predictor** (`src/neural_predictor.py`) — an
   `MLPRegressor` trained to predict a tour's total distance from a
   fixed-length feature vector (route count, mean/std route length,
   consecutive-customer distance statistics, route spread, capacity
   utilization) instead of the raw permutation, which a feedforward
   network can't use directly.

3. **Surrogate-accelerated GA** — each generation generates a 3x larger
   candidate pool via crossover/mutation, ranks the whole pool with the
   cheap neural predictor (batched, not looped), keeps the top 70% by
   predicted quality plus a random 30% of the remainder (exploration
   safety net against surrogate bias), and only spends exact evaluations
   on the survivors. **Both GA variants use an identical exact-evaluation
   budget** (`pop_size x generations = 9000`) — the comparison is
   solution quality at equal cost, not evaluation count.

## Results

| | Best distance | Wall-clock time |
|---|---|---|
| Baseline GA | **1080.85** | 1.15s |
| GA + Neural Predictor | 1182.59 | 8.53s |

The surrogate-accelerated GA finished **9.4% worse** than the plain
baseline on this instance, despite the predictor's own validation
accuracy looking very good (MAPE 0.92%).

### Why — and what this taught me

This wasn't the result I expected going in, and the debugging process is
the more interesting part of this project:

1. **First attempt was 35% worse.** The surrogate was trained only on
   uniformly random tours. Random permutations all look statistically
   similar (uniformly poor structure), but the GA's actual population
   evolves toward much more organized tours over 150 generations — a
   different feature distribution than training ever covered. **Fix**:
   mix training data with samples from a short pilot GA run spanning
   multiple quality levels. This alone improved the gap to 18.5%.

2. **Second issue: an 11x slowdown.** The surrogate GA took 37s vs. the
   baseline's 3.4s — defeating the point of a "cheap" surrogate. The
   cause was calling `model.predict()` once per individual in a Python
   loop; `MLPRegressor`'s per-call overhead dominated. **Fix**: batch
   all candidates in the pool into a single `predict()` call
   (`predict_batch` in `neural_predictor.py`). This cut runtime to
   ~8-11s while also slightly improving quality (fewer redundant
   feature-extraction passes, same logic).

3. **Third issue, and the real remaining limitation**: pure top-K
   selection by predicted rank discards any candidate the model
   slightly underrates, with no chance to prove itself. Adding 30%
   random retention from the non-top-ranked pool (a standard
   exploration safety net for noisy surrogates) improved the gap from
   18.5% to 9.4%.

4. **Root cause of the remaining 9.4% gap**, found by plotting the
   validation scatter (`results/predictor_accuracy.png`) against the
   actual convergence curve (`results/convergence_comparison.png`): the
   predictor's training data (random + short-pilot-GA tours) spans
   **~1600-2400** in tour distance, but the *real* 150-generation GA run
   converges down to **~1080-1200** — well below anything the predictor
   was trained on. The convergence plot confirms this exactly: the two
   GA variants track each other closely for the first ~2000 evaluations,
   then diverge as the baseline keeps refining into a quality range the
   surrogate has never seen and can't rank reliably.

**What would actually fix this** (noted as future work rather than
implemented here, given time constraints): periodically retrain the
surrogate during the GA run using tours sampled from the *current*
population, so its training distribution tracks the search instead of
being fixed upfront. This is closer to what the literature on
surrogate-assisted metaheuristics actually does, and is the natural next
step for this project.

## Files

```
src/
  vrp_instance.py       Instance generation + TSPLIB/CVRPLIB .vrp I/O
  genetic_algorithm.py  Baseline GA (encoding, operators, exact fitness)
  neural_predictor.py   Feature extraction, MLP training, surrogate GA
  run_experiment.py     Full pipeline: run both, compare, plot, save
data/
  synth_n40.vrp          Generated CVRP instance (standard format)
results/
  convergence_comparison.png   Solution quality vs. exact evals spent
  predictor_accuracy.png       NN predicted vs. actual tour distance
  best_routes.png               Visualized best solution found
  summary.json                  All numeric results
```

## Running it

```
python src/run_experiment.py
```

Runs in well under a minute on a CPU-only machine (no GPU used or
needed anywhere in this project).

## Skills demonstrated

- Genetic algorithm design for a constrained combinatorial problem (CVRP)
- Feature engineering for a neural surrogate model on non-fixed-length
  combinatorial objects
- Diagnosing and fixing a real ML generalization failure (train/deploy
  distribution mismatch) using evidence from the actual plots, not
  guesswork
- Performance debugging (identifying and fixing an 11x slowdown from
  unbatched inference)
- Honest empirical reporting: this project reports a genuine negative
  result with a precise, evidenced explanation, rather than a fabricated
  success — the debugging narrative here is a more accurate signal of
  research ability than a clean "it worked" would have been.
