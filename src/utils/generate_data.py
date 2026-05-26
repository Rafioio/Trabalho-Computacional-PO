#!/usr/bin/env python3
"""
Random data generator for the SouBuz bus stop optimization model.

Generates synthetic instances in JSON format for the Gurobi model.

Usage:
    python generate_data.py                                   # default: small instance
    python generate_data.py --num-n 50 --num-k 3 --num-q 5
    python generate_data.py --output mydata.json --seed 99
"""

import argparse
import json
import math
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Generate random SouBuz data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-n", type=int, default=50, help="Number of nodes")
    parser.add_argument("--num-k", type=int, default=3, help="Number of routes")
    parser.add_argument("--num-q", type=int, default=5, help="Number of demand zones")
    parser.add_argument("--num-c", type=int, default=None, help="Number of centroids (default: ~8%% of N)")
    parser.add_argument("--route-len", type=int, default=None, help="Nodes per route (default: ~40%% of N)")
    parser.add_argument("--d-route-max", type=float, default=250.0, help="Max route spacing (m)")
    parser.add_argument("--d-walk-max", type=float, default=100.0, help="Max walk distance (m)")
    parser.add_argument("--capt", type=int, default=8000, help="Base system capacity")
    parser.add_argument("--m-max", type=int, default=7, help="Max routes per stop")
    parser.add_argument("--P", type=float, default=1000.0, help="Penalty for unserved demand")
    parser.add_argument("--omega", type=float, default=1.0, help="Additional capacity cost weight")
    parser.add_argument("--output", default="dados_generated.json", help="Output .json file")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    NumN = args.num_n
    NumK = args.num_k
    NumQ = args.num_q
    NumC = args.num_c if args.num_c is not None else max(1, NumN // 12)
    route_len = args.route_len if args.route_len is not None else max(2, NumN * 2 // 5)

    positions = [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(NumN)]

    all_nodes = list(range(1, NumN + 1))
    rng.shuffle(all_nodes)
    C = sorted(all_nodes[:NumC])
    T = sorted(all_nodes[NumC:])

    de = [round(rng.uniform(100, 500)) for _ in range(NumQ)]
    w = [round(rng.uniform(0.5, 1.0), 2) for _ in range(NumN)]

    demand_positions = [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(NumQ)]
    d = []
    for q in range(NumQ):
        row = [
            round(math.sqrt((demand_positions[q][0] - positions[n][0]) ** 2 + (demand_positions[q][1] - positions[n][1]) ** 2), 2)
            for n in range(NumN)
        ]
        d.append(row)

    V = []
    V_tamanho = []
    I_set = set()
    for k in range(1, NumK + 1):
        nodes = all_nodes.copy()
        rng.shuffle(nodes)
        route = nodes[:route_len]
        V.append(route)
        V_tamanho.append(len(route))
        for n in route:
            I_set.add((n, k))

    L_set = set()
    for q in range(1, NumQ + 1):
        selected = rng.sample(range(1, NumK + 1), rng.randint(2, max(2, NumK)))
        for k in selected:
            L_set.add((q, k))

    I = sorted(I_set, key=lambda x: (x[1], x[0]))
    L = sorted(L_set, key=lambda x: (x[1], x[0]))

    D = []
    for route in V:
        Dk = []
        for n1_idx in range(NumN):
            row = [0.0] * NumN
            for n2 in route:
                p1 = positions[n1_idx]
                p2 = positions[n2 - 1]
                row[n2 - 1] = round(math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), 2)
            Dk.append(row)
        D.append(Dk)

    json_data = {
        "Q": list(range(1, NumQ + 1)),
        "N": list(range(1, NumN + 1)),
        "K": list(range(1, NumK + 1)),
        "C": C,
        "T": T,
        "I": I,
        "L": L,
        "de": de,
        "d": d,
        "w": w,
        "D": D,
        "V": V,
        "V_tamanho": V_tamanho,
        "P": args.P,
        "omega": args.omega,
        "W1": 1e-3,
        "W2": 1.0,
        "W3": 0.94,
        "W4": 1.0,
        "Capt": args.capt,
        "m_max": args.m_max,
        "d_route_max": args.d_route_max,
        "d_walk_max": args.d_walk_max,
    }

    with open(args.output, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()