"""
CVRP Instance Generation and Loading
======================================

Generates a synthetic but TSPLIB/CVRPLIB-style Capacitated Vehicle Routing
Problem (CVRP) instance: one depot, N customers with demand, vehicles of
fixed capacity, unlimited fleet size (minimize total distance across all
routes needed to serve every customer within capacity).

A fixed random seed is used so results are reproducible. The .vrp writer
follows the standard TSPLIB CVRP format, so a real CVRPLIB instance file
can be dropped in later and loaded with `load_vrp_file` instead, without
changing any downstream code.
"""

import numpy as np
import os

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")


def generate_instance(n_customers=40, capacity=100, seed=42, grid_size=100):
    """Generate a synthetic CVRP instance with a fixed seed.

    Returns:
        coords: (n_customers+1, 2) array, index 0 = depot
        demands: (n_customers+1,) array, demand[0] = 0 (depot)
        capacity: vehicle capacity (int)
    """
    rng = np.random.default_rng(seed)
    n_nodes = n_customers + 1

    coords = rng.uniform(0, grid_size, size=(n_nodes, 2))
    coords[0] = [grid_size / 2, grid_size / 2]  # depot at center

    demands = rng.integers(5, 30, size=n_nodes)
    demands[0] = 0  # depot has no demand

    return coords, demands, capacity


def save_vrp_file(coords, demands, capacity, path, name="SYNTH-n40"):
    """Save in standard TSPLIB/CVRPLIB .vrp format."""
    n = len(coords)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"NAME : {name}\n")
        f.write("TYPE : CVRP\n")
        f.write(f"DIMENSION : {n}\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write(f"CAPACITY : {capacity}\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(coords, start=1):
            f.write(f"{i} {x:.4f} {y:.4f}\n")
        f.write("DEMAND_SECTION\n")
        for i, d in enumerate(demands, start=1):
            f.write(f"{i} {int(d)}\n")
        f.write("DEPOT_SECTION\n1\n-1\nEOF\n")


def load_vrp_file(path):
    """Load a standard TSPLIB/CVRPLIB .vrp format instance file."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines()]

    capacity = None
    dimension = None
    coords = {}
    demands = {}
    section = None

    for line in lines:
        if line.startswith("CAPACITY"):
            capacity = int(line.split(":")[-1].strip())
        elif line.startswith("DIMENSION"):
            dimension = int(line.split(":")[-1].strip())
        elif line.startswith("NODE_COORD_SECTION"):
            section = "coord"
            continue
        elif line.startswith("DEMAND_SECTION"):
            section = "demand"
            continue
        elif line.startswith("DEPOT_SECTION") or line.startswith("EOF"):
            section = None
            continue

        if section == "coord" and line:
            parts = line.split()
            if len(parts) >= 3:
                idx = int(parts[0])
                coords[idx] = (float(parts[1]), float(parts[2]))
        elif section == "demand" and line:
            parts = line.split()
            if len(parts) >= 2:
                idx = int(parts[0])
                demands[idx] = int(parts[1])

    n = dimension
    coords_arr = np.array([coords[i] for i in range(1, n + 1)])
    demands_arr = np.array([demands[i] for i in range(1, n + 1)])
    return coords_arr, demands_arr, capacity


def distance_matrix(coords):
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


if __name__ == "__main__":
    coords, demands, capacity = generate_instance()
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "synth_n40.vrp")
    save_vrp_file(coords, demands, capacity, out_path)
    print(f"Generated instance: {len(coords)-1} customers, capacity={capacity}")
    print(f"Total demand: {demands.sum()}, min vehicles needed: {int(np.ceil(demands.sum()/capacity))}")
    print(f"Saved to: {out_path}")
