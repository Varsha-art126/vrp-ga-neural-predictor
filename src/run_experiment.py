"""
Full Experiment: GA baseline vs. GA + Neural Cost Predictor on CVRP
======================================================================

1. Generate (or load) a CVRP instance.
2. Run the baseline GA (exact fitness every individual, every generation).
3. Build a training set of (tour features -> exact cost) pairs and train
   an MLP surrogate cost predictor.
4. Run the surrogate-accelerated GA (NN ranks a larger candidate pool,
   only the best survive to expensive exact evaluation).
5. Compare: solution quality vs. number of EXACT evaluations spent
   (the real cost driver in practice), plus wall-clock time.
6. Save plots + a results summary to ../results/.
"""

import os
import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vrp_instance import generate_instance, save_vrp_file, distance_matrix
from genetic_algorithm import run_ga, decode_routes, route_distance
from neural_predictor import build_training_set, train_predictor, run_ga_with_surrogate

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")
RESULTS_DIR = os.path.join(HERE, "..", "results")

GENERATIONS = 150
POP_SIZE = 60
SEED = 42


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("1. GENERATING CVRP INSTANCE")
    print("=" * 60)
    coords, demands, capacity = generate_instance(n_customers=40, capacity=100, seed=SEED)
    save_vrp_file(coords, demands, capacity, os.path.join(DATA_DIR, "synth_n40.vrp"))
    dist_matrix = distance_matrix(coords)
    min_vehicles = int(np.ceil(demands.sum() / capacity))
    print(f"  {len(coords)-1} customers, capacity={capacity}, "
          f"total demand={demands.sum()}, min vehicles~{min_vehicles}")

    print("\n" + "=" * 60)
    print("2. BASELINE GA (exact fitness every individual)")
    print("=" * 60)
    t0 = time.time()
    best_tour_base, best_fit_base, history_base = run_ga(
        demands, capacity, dist_matrix,
        pop_size=POP_SIZE, generations=GENERATIONS, seed=SEED,
    )
    time_base = time.time() - t0
    evals_base = history_base[-1][1]
    print(f"  Best distance: {best_fit_base:.2f}")
    print(f"  Exact evaluations used: {evals_base}")
    print(f"  Wall-clock time: {time_base:.2f}s")

    print("\n" + "=" * 60)
    print("3. TRAINING NEURAL COST PREDICTOR")
    print("=" * 60)
    n_train_samples = 800
    X, y = build_training_set(n_train_samples, demands, capacity, dist_matrix, seed=SEED + 1)
    model, scaler, metrics = train_predictor(X, y, seed=SEED)
    print(f"  Trained on {n_train_samples} sampled tours")
    print(f"  Validation MAE: {metrics['val_mae']:.2f}")
    print(f"  Validation MAPE: {metrics['val_mape']:.2f}%")

    print("\n" + "=" * 60)
    print("4. SURROGATE-ACCELERATED GA (NN pre-filters candidate pool)")
    print("=" * 60)
    t0 = time.time()
    best_tour_nn, best_fit_nn, history_nn = run_ga_with_surrogate(
        demands, capacity, dist_matrix, model, scaler,
        pop_size=POP_SIZE, generations=GENERATIONS, pool_multiplier=3, seed=SEED,
    )
    time_nn = time.time() - t0
    evals_nn = history_nn[-1][1]
    print(f"  Best distance: {best_fit_nn:.2f}")
    print(f"  Exact evaluations used: {evals_nn}")
    print(f"  Wall-clock time: {time_nn:.2f}s")

    print("\n" + "=" * 60)
    print("5. COMPARISON (both use the SAME exact-eval budget: "
          f"{POP_SIZE} x {GENERATIONS} = {POP_SIZE*GENERATIONS})")
    print("=" * 60)
    quality_gap = (best_fit_nn - best_fit_base) / best_fit_base * 100
    print(f"  Baseline GA:            distance={best_fit_base:.2f}, time={time_base:.2f}s")
    print(f"  GA + Neural Predictor:  distance={best_fit_nn:.2f}, time={time_nn:.2f}s")
    print(f"  Solution quality gap at EQUAL exact-eval budget: {quality_gap:+.2f}% "
          f"({'worse' if quality_gap > 0 else 'better'} than baseline)")
    print(f"  (The surrogate spends the same exact-eval budget, but pre-filters a "
          f"{3}x larger candidate pool per generation via cheap NN ranking - the "
          f"question is whether that pre-filtering finds better candidates than "
          f"plain tournament selection would, at no extra exact-eval cost.)")

    # ---- Plot 1: convergence vs. exact evaluations spent ----
    fig, ax = plt.subplots(figsize=(7, 5))
    fits_b, evals_b = zip(*history_base)
    fits_n, evals_n = zip(*history_nn)
    ax.plot(evals_b, fits_b, label="Baseline GA", linewidth=2)
    ax.plot(evals_n, fits_n, label="GA + Neural Predictor", linewidth=2)
    ax.set_xlabel("Number of EXACT fitness evaluations")
    ax.set_ylabel("Best distance found so far")
    ax.set_title("Convergence: Solution Quality vs. Exact Evaluations Spent")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "convergence_comparison.png"), dpi=150)
    plt.close(fig)

    # ---- Plot 2: predicted vs actual cost (surrogate quality) ----
    from sklearn.model_selection import train_test_split
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
    X_val_s = scaler.transform(X_val)
    y_pred = model.predict(X_val_s)
    fig2, ax2 = plt.subplots(figsize=(5.5, 5.5))
    ax2.scatter(y_val, y_pred, alpha=0.5, s=15)
    lims = [min(y_val.min(), y_pred.min()), max(y_val.max(), y_pred.max())]
    ax2.plot(lims, lims, "r--", label="Perfect prediction")
    ax2.set_xlabel("Actual tour distance")
    ax2.set_ylabel("NN-predicted tour distance")
    ax2.set_title(f"Neural Cost Predictor Accuracy\nMAPE = {metrics['val_mape']:.2f}%")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "predictor_accuracy.png"), dpi=150)
    plt.close(fig2)

    # ---- Plot 3: best routes found (baseline) ----
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    routes = decode_routes(best_tour_base, demands, capacity)
    colors = plt.cm.tab10(np.linspace(0, 1, len(routes)))
    ax3.scatter(*coords[0], c="black", marker="s", s=100, label="Depot", zorder=5)
    for r, c in zip(routes, colors):
        full = [0] + r + [0]
        pts = coords[full]
        ax3.plot(pts[:, 0], pts[:, 1], "-o", color=c, markersize=5)
    ax3.set_title(f"Best Solution Found (Baseline GA)\nTotal distance = {best_fit_base:.1f}")
    ax3.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "best_routes.png"), dpi=150)
    plt.close(fig3)

    # ---- Save numeric summary ----
    summary = {
        "instance": {
            "n_customers": len(coords) - 1,
            "capacity": int(capacity),
            "total_demand": int(demands.sum()),
            "min_vehicles": min_vehicles,
        },
        "baseline_ga": {
            "best_distance": float(best_fit_base),
            "exact_evaluations": int(evals_base),
            "wall_clock_seconds": float(time_base),
            "n_routes": len(decode_routes(best_tour_base, demands, capacity)),
        },
        "surrogate_ga": {
            "best_distance": float(best_fit_nn),
            "exact_evaluations": int(evals_nn),
            "wall_clock_seconds": float(time_nn),
            "n_routes": len(decode_routes(best_tour_nn, demands, capacity)),
        },
        "predictor": {
            "n_training_samples": n_train_samples,
            "val_mae": float(metrics["val_mae"]),
            "val_mape_pct": float(metrics["val_mape"]),
        },
        "comparison": {
            "quality_gap_pct": float(quality_gap),
            "note": "Both GA variants use an identical exact-eval budget "
                    "(pop_size x generations); the surrogate pre-filters a "
                    "larger candidate pool per generation instead of reducing "
                    "total exact evaluations.",
        },
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved plots and summary.json to: {RESULTS_DIR}")
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
