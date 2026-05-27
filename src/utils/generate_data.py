#!/usr/bin/env python3
"""
Synthetic data generator for the SouBuz bus stop optimization model.

Merges the realistic city generation from the 'entradas' branch (clustered
positions, log-normal demand, beta-distributed quality, minimum-distance
enforcement) with the route generation required by the Gurobi solver.

Uses numpy for vectorized distance matrix computation and random generation.

Output is a flat JSON file compatible with loader.load_json().

Usage:
    python src/utils/generate_data.py                              # 50 nodes, 5 routes, 35 zones
    python src/utils/generate_data.py --num-n 500 --num-k 10 --num-q 30
    python src/utils/generate_data.py --seed 99 --output mydata.json
    python src/utils/generate_data.py --scenarios 10               # batch mode
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np


DEFAULT_NUM_N = 50
DEFAULT_NUM_K = 5
DEFAULT_NUM_Q = 35
DEFAULT_GRID_WIDTH = 3000.0
DEFAULT_GRID_HEIGHT = 3000.0
DEFAULT_MIN_DIST = 80.0
DEFAULT_D_WALK_MAX = 500.0
DEFAULT_D_ROUTE_MAX = 800.0
DEFAULT_P = 1000.0
DEFAULT_CAPT = 800
DEFAULT_M_MAX = 3
DEFAULT_OMEGA = 50.0
DEFAULT_W1 = 0.35
DEFAULT_W2 = 0.15
DEFAULT_W3 = 0.30
DEFAULT_W4 = 0.20
DEFAULT_ROUTE_LEN_FRAC = 0.35


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate synthetic SouBuz data")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--num-n", type=int, default=DEFAULT_NUM_N, help="Number of nodes")
    parser.add_argument("--num-k", type=int, default=DEFAULT_NUM_K, help="Number of routes")
    parser.add_argument("--num-q", type=int, default=DEFAULT_NUM_Q, help="Number of demand zones")
    parser.add_argument("--num-c", type=int, default=None, help="Centroids (default: ~8%% of N)")
    parser.add_argument("--route-len", type=int, default=None, help="Nodes per route (default: ~35%% of N)")
    parser.add_argument("--grid-width", type=float, default=DEFAULT_GRID_WIDTH, help="City grid width (m)")
    parser.add_argument("--grid-height", type=float, default=DEFAULT_GRID_HEIGHT, help="City grid height (m)")
    parser.add_argument("--min-dist", type=float, default=DEFAULT_MIN_DIST, help="Minimum distance between nodes (m)")
    parser.add_argument("--d-route-max", type=float, default=DEFAULT_D_ROUTE_MAX, help="Max route spacing (m)")
    parser.add_argument("--d-walk-max", type=float, default=DEFAULT_D_WALK_MAX, help="Max walk distance (m)")
    parser.add_argument("--capt", type=int, default=DEFAULT_CAPT, help="Base system capacity")
    parser.add_argument("--m-max", type=int, default=DEFAULT_M_MAX, help="Max routes per stop")
    parser.add_argument("--P", type=float, default=DEFAULT_P, help="Penalty for unserved demand")
    parser.add_argument("--omega", type=float, default=DEFAULT_OMEGA, help="Additional capacity cost weight")
    parser.add_argument("--W1", type=float, default=DEFAULT_W1, help="Weight for f1 (social cost)")
    parser.add_argument("--W2", type=float, default=DEFAULT_W2, help="Weight for f2 (technical feasibility)")
    parser.add_argument("--W3", type=float, default=DEFAULT_W3, help="Weight for f3 (infrastructure cost)")
    parser.add_argument("--W4", type=float, default=DEFAULT_W4, help="Weight for f4 (spacing penalty)")
    parser.add_argument("--output", default=None, help="Output .json file")
    parser.add_argument("--scenarios", type=int, default=None, help="Batch-generate N scenarios")
    parser.add_argument("--prefix", default="cenario", help="Prefix for batch output files")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return parser.parse_args(argv)


def generate_positions(num_n, grid_w, grid_h, min_dist, nprng):
    positions_list = []
    for _ in range(num_n):
        for attempt in range(2000):
            if nprng.random() < 0.6 and positions_list:
                base = positions_list[nprng.integers(len(positions_list))]
                candidate = base + nprng.normal(0, [grid_w * 0.08, grid_h * 0.08])
            else:
                candidate = nprng.uniform(0, [grid_w, grid_h])
            candidate = np.clip(candidate, 0, [grid_w, grid_h])
            if not positions_list:
                positions_list.append(candidate)
                break
            if np.all(np.linalg.norm(np.array(positions_list) - candidate, axis=1) >= min_dist):
                positions_list.append(candidate)
                break
        else:
            positions_list.append(nprng.uniform(0, [grid_w, grid_h]))
    return np.array(positions_list)


def generate_demand_positions(positions, num_q, grid_w, grid_h, d_walk_max, nprng):
    num_n = len(positions)
    idxs = nprng.integers(num_n, size=num_q)
    use_cluster = nprng.random(size=num_q) < 0.8
    angles = nprng.uniform(0, 2 * np.pi, size=num_q)
    radii = nprng.uniform(0, d_walk_max * 0.7, size=num_q)
    demand_positions = np.empty((num_q, 2))
    for i in range(num_q):
        if use_cluster[i]:
            base = positions[idxs[i]]
            demand_positions[i] = base + radii[i] * np.array([np.cos(angles[i]), np.sin(angles[i])])
        else:
            demand_positions[i] = nprng.uniform(0, [grid_w, grid_h])
    np.clip(demand_positions, 0, [grid_w, grid_h], out=demand_positions)
    demand_levels = np.round(nprng.lognormal(4.0, 0.8, size=num_q)).astype(int)
    np.clip(demand_levels, 5, 200, out=demand_levels)
    return demand_positions, demand_levels.tolist()


def generate_quality(num_n, nprng):
    q = nprng.beta(2, 2, size=num_n)
    return np.round(0.2 + 0.8 * q, 6).tolist()


def build_route_matrices(positions, all_route_indices):
    NumN = positions.shape[0]
    D_list = []
    for route_idxs in all_route_indices:
        route_pos = positions[route_idxs]
        diffs = positions[:, np.newaxis, :] - route_pos[np.newaxis, :, :]
        dists = np.round(np.linalg.norm(diffs, axis=2), 2)
        Dk = np.zeros((NumN, NumN))
        Dk[:, route_idxs] = dists
        D_list.append(Dk.tolist())
    return D_list


def compute_d_matrix(demand_positions, node_positions):
    diffs = demand_positions[:, np.newaxis, :] - node_positions[np.newaxis, :, :]
    return np.round(np.linalg.norm(diffs, axis=2), 2).tolist()


def compute_statistics(d, de, w, d_walk_max):
    d_arr = np.array(d)
    accessible = d_arr[d_arr <= d_walk_max]
    de_arr = np.array(de)
    w_arr = np.array(w)
    return {
        "perc_acessiveis": round(len(accessible) / d_arr.size * 100, 2) if d_arr.size > 0 else 0,
        "dist_media_acessiveis": round(float(np.mean(accessible)), 2) if len(accessible) > 0 else 0,
        "dist_minima": round(float(np.min(d_arr)), 2) if d_arr.size > 0 else 0,
        "dist_maxima": round(float(np.max(d_arr)), 2) if d_arr.size > 0 else 0,
        "demanda_total": int(np.sum(de_arr)),
        "demanda_media": round(float(np.mean(de_arr)), 2) if len(de_arr) > 0 else 0,
        "qualidade_media": round(float(np.mean(w_arr)), 6) if len(w_arr) > 0 else 0,
        "qualidade_min": round(float(np.min(w_arr)), 6) if len(w_arr) > 0 else 0,
        "qualidade_max": round(float(np.max(w_arr)), 6) if len(w_arr) > 0 else 0,
    }


def generate_single(args, seed, nprng, quiet=False):
    NumN = args.num_n
    NumK = args.num_k
    NumQ = args.num_q
    NumC = args.num_c if args.num_c is not None else max(1, NumN // 12)
    route_len = args.route_len if args.route_len is not None else max(2, int(NumN * DEFAULT_ROUTE_LEN_FRAC))
    grid_w = args.grid_width
    grid_h = args.grid_height
    min_dist = args.min_dist
    d_walk_max = args.d_walk_max

    node_positions = generate_positions(NumN, grid_w, grid_h, min_dist, nprng)

    all_nodes = list(range(1, NumN + 1))
    nprng.shuffle(all_nodes)
    C = sorted(all_nodes[:NumC])
    T = sorted(all_nodes[NumC:])

    demand_positions, de = generate_demand_positions(node_positions, NumQ, grid_w, grid_h, d_walk_max, nprng)
    w = generate_quality(NumN, nprng)
    d = compute_d_matrix(demand_positions, node_positions)

    V = []
    V_tamanho = []
    I_set = set()
    route_node_indices = []
    for k in range(NumK):
        perm = nprng.permuted(np.arange(NumN))
        sel = perm[:route_len]
        sel_sorted = np.sort(sel[np.argsort(node_positions[sel, 0])])
        route_node_indices.append(sel_sorted)
        route = (sel_sorted + 1).tolist()
        V.append(route)
        V_tamanho.append(len(route))
        for n in route:
            I_set.add((n, k + 1))
    I = sorted(I_set, key=lambda x: (x[1], x[0]))

    L_set = set()
    for q in range(1, NumQ + 1):
        k_count = max(2, min(NumK, nprng.integers(2, NumK + 1)))
        selected = nprng.choice(range(1, NumK + 1), size=k_count, replace=False)
        for k in selected:
            L_set.add((q, int(k)))
    L = sorted(L_set, key=lambda x: (x[1], x[0]))

    D = build_route_matrices(node_positions, route_node_indices)
    stats = compute_statistics(d, de, w, d_walk_max)

    data = {
        "NumN": NumN,
        "NumK": NumK,
        "NumQ": NumQ,
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
        "W1": args.W1,
        "W2": args.W2,
        "W3": args.W3,
        "W4": args.W4,
        "Capt": args.capt,
        "m_max": args.m_max,
        "d_route_max": args.d_route_max,
        "d_walk_max": d_walk_max,
        "metadata": {
            "versao": "2.0",
            "data_geracao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "semente": seed,
            "descricao": "Dados sinteticos SouBuz (gerados via src/utils/generate_data.py)",
        },
        "visualizacao": {
            "posicoes_pontos": [[round(float(x), 2), round(float(y), 2)] for x, y in node_positions],
            "posicoes_demandas": [[round(float(x), 2), round(float(y), 2)] for x, y in demand_positions],
            "qualidades_pontos": w,
            "niveis_demanda": de,
        },
        "estatisticas": stats,
    }

    if not quiet:
        print(f"Gereados: {NumN} nos, {NumK} rotas, {NumQ} zonas de demanda (semente={seed})")
        print(f"  Demanda total={stats['demanda_total']}, acessiveis={stats['perc_acessiveis']:.1f}%")
        print(f"  Qualidade media={stats['qualidade_media']:.3f}")

    return data


def generate_scenarios(args, nprng):
    scenarios = []
    for i in range(1, args.scenarios + 1):
        seed = int(nprng.integers(0, 2**31 - 1))
        scenario_rng = np.random.default_rng(seed)
        data = generate_single(args, seed, scenario_rng, quiet=args.quiet)
        filename = f"{args.prefix}_{i:02d}.json"
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        scenarios.append({
            "cenario": i,
            "arquivo": filename,
            "semente": seed,
            "demanda_total": data["estatisticas"]["demanda_total"],
        })
        if not args.quiet:
            print(f"  [{i}/{args.scenarios}] {filename}")
    resumo = os.path.join(os.path.dirname(scenarios[0]["arquivo"]) if scenarios else ".", "resumo_cenarios.json")
    with open(resumo, "w") as f:
        json.dump(scenarios, f, indent=2)
    if not args.quiet:
        print(f"Resumo salvo em {resumo}")


def main(argv=None):
    args = parse_args(argv)

    if args.seed is not None:
        seed = args.seed
        nprng = np.random.default_rng(seed)
        if not args.quiet:
            print(f"[generate_data] Semente fixa: {seed}")
    else:
        seed = int(datetime.now().timestamp() * 1_000_000) % (2**31)
        nprng = np.random.default_rng(seed)
        if not args.quiet:
            print(f"[generate_data] Semente aleatoria: {seed}")

    if args.scenarios:
        generate_scenarios(args, nprng)
        return

    data = generate_single(args, seed, nprng, quiet=args.quiet)

    output = args.output
    if output is None:
        output = "dados_generated.json"
    with open(output, "w") as f:
        json.dump(data, f, indent=2)
    if not args.quiet:
        print(f"Escrito: {output} ({os.path.getsize(output) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()