#!/usr/bin/env python3
"""
Convert BHBus CSV data to SouBuz model input JSON.

Reads the BH bus stop CSV (with UTM coordinates) and produces a JSON file
compatible with src/model/solver.py.

Usage:
    python src/utils/convert_csv_to_data.py \\
        --csv src/data/20260504_ponto_onibus.csv \\
        --output dados_bh.json

    # Limit to first N routes and max stops per route for smaller instances:
    python src/utils/convert_csv_to_data.py \\
        --max-routes 5 --max-stops 50 --output small_bh.json
"""

import argparse
import csv
import json
import math
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Convert BH bus CSV to SouBuz input JSON")
    parser.add_argument("--csv", default="src/data/20260504_ponto_onibus.csv",
                        help="Path to BH bus stop CSV")
    parser.add_argument("--output", "-o", default="dados_bh.json",
                        help="Output JSON file path")
    parser.add_argument("--max-routes", type=int, default=None,
                        help="Max number of routes to include (default: all)")
    parser.add_argument("--max-stops", type=int, default=None,
                        help="Max stops per route (default: all)")
    parser.add_argument("--num-q", type=int, default=30,
                        help="Number of demand zones to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for demand/quality generation")
    parser.add_argument("--d-route-max", type=float, default=1000.0,
                        help="Max spacing between consecutive stops (m)")
    parser.add_argument("--d-walk-max", type=float, default=500.0,
                        help="Max walking distance (m)")
    parser.add_argument("--capt", type=int, default=800,
                        help="Base system capacity")
    parser.add_argument("--P", type=float, default=1000.0,
                        help="Penalty for unserved demand")
    return parser.parse_args()


def parse_point(geom: str):
    """Parse 'POINT (x y)' string to (x, y) float tuple."""
    geom = geom.strip().replace("POINT (", "").replace(")", "")
    x_str, y_str = geom.split()
    return round(float(x_str), 2), round(float(y_str), 2)


def deduplicate_stops(routes_dict):
    """
    Merge stops that share the same coordinates across different routes.
    Returns (nodes, stop_id_map) where stop_id_map maps old ID -> new node index.
    """
    coord_to_idx = {}
    node_counter = 1

    for route_key, stops in routes_dict.items():
        for entry in stops:
            coord = entry["coord"]
            if coord not in coord_to_idx:
                coord_to_idx[coord] = node_counter
                node_counter += 1

    return coord_to_idx


def generate_demand_zones(node_positions, num_q, rng, d_walk_max, grid_padding=500):
    """Generate demand zones clustered around existing nodes."""
    positions = np.array(node_positions)
    min_x, min_y = positions.min(axis=0) - grid_padding
    max_x, max_y = positions.max(axis=0) + grid_padding

    demand_positions = []
    demand_levels = []

    for _ in range(num_q):
        if rng.random() < 0.8 and len(positions) > 0:
            base = positions[rng.integers(len(positions))]
            angle = rng.uniform(0, 2 * math.pi)
            radius = rng.uniform(0, d_walk_max * 0.7)
            dx = base[0] + radius * math.cos(angle)
            dy = base[1] + radius * math.sin(angle)
        else:
            dx = rng.uniform(min_x, max_x)
            dy = rng.uniform(min_y, max_y)

        demand_positions.append((round(dx, 2), round(dy, 2)))
        demand_levels.append(int(round(rng.lognormal(4.0, 0.8))))

    demand_levels = [max(5, min(200, d)) for d in demand_levels]
    return demand_positions, demand_levels


def compute_walking_distances(demand_positions, node_positions):
    """Euclidean distance matrix between demand zones and nodes."""
    d_arr = np.array(demand_positions)[:, np.newaxis, :]
    n_arr = np.array(node_positions)[np.newaxis, :, :]
    dists = np.round(np.linalg.norm(d_arr - n_arr, axis=2), 2)
    return dists.tolist()


def build_route_distance_matrices(node_positions, routes):
    """Build D[k][i][j] for each route: distance between any two nodes."""
    num_n = len(node_positions)
    pos = np.array(node_positions)
    D_list = []

    for route in routes:
        route_idxs = [n - 1 for n in route]
        route_pos = pos[route_idxs]
        diffs = pos[:, np.newaxis, :] - route_pos[np.newaxis, :, :]
        dists = np.round(np.linalg.norm(diffs, axis=2), 2)
        Dk = np.zeros((num_n, num_n))
        Dk[:, route_idxs] = dists
        D_list.append(Dk.tolist())

    return D_list


def generate_quality(num_n, rng):
    """Generate technical quality scores (0.2–1.0) using beta distribution."""
    q = rng.beta(2, 2, size=num_n)
    return [round(0.2 + 0.8 * v, 6) for v in q.tolist()]


def build_L_set(num_q, num_k, rng):
    """Build demand-route pairs (q, k)."""
    L = []
    for q in range(1, num_q + 1):
        k_count = max(1, min(num_k, rng.integers(1, num_k + 1)))
        selected = rng.choice(range(1, num_k + 1), size=k_count, replace=False)
        for k in selected:
            L.append((q, int(k)))
    return sorted(set(L), key=lambda x: (x[1], x[0]))


def build_C_set(num_n, rng):
    """Build centroids (subset of N for mandatory activation)."""
    all_nodes = list(range(1, num_n + 1))
    rng.shuffle(all_nodes)
    num_c = max(1, num_n // 12)
    return sorted(all_nodes[:num_c])


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # ---- Read CSV ----
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return 1

    print(f"Reading CSV: {csv_path}")
    raw_routes = OrderedDict()

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            route_key = (row["COD_LINHA"], row["NOME_SUB_LINHA"])
            coord = parse_point(row["GEOMETRIA"])
            stop_id = row["IDENTIFICADOR_PONTO_ONIBUS"]

            if route_key not in raw_routes:
                raw_routes[route_key] = []
            raw_routes[route_key].append({
                "id": stop_id,
                "coord": coord,
            })

    print(f"Read {len(raw_routes)} route variants from CSV")

    # ---- Filter routes ----
    route_keys = list(raw_routes.keys())
    if args.max_routes and args.max_routes < len(route_keys):
        rng.shuffle(route_keys)
        route_keys = route_keys[:args.max_routes]

    # ---- Deduplicate stops by coordinate ----
    selected_routes_list = {k: raw_routes[k] for k in route_keys}
    coord_to_node = deduplicate_stops(selected_routes_list)

    node_positions = [None] * (len(coord_to_node) + 1)
    for (x, y), idx in coord_to_node.items():
        node_positions[idx] = [x, y]

    num_n = len(node_positions) - 1
    num_k = len(route_keys)
    print(f"Nodes (after dedup): {num_n}, Routes: {num_k}")

    # ---- Build routes (V) and route-stop pairs (I) ----
    V = []
    V_tamanho = []
    I_set = set()
    route_name_map = {}

    for k_idx, key in enumerate(route_keys):
        k = k_idx + 1
        stops = selected_routes_list[key]

        # Map each stop to its deduplicated node index
        route_nodes = []
        for entry in stops:
            node_id = coord_to_node[entry["coord"]]
            route_nodes.append(node_id)
            I_set.add((node_id, k))

        route_nodes_ordered = list(OrderedDict.fromkeys(route_nodes))

        if args.max_stops and len(route_nodes_ordered) > args.max_stops:
            route_nodes_ordered = route_nodes_ordered[:args.max_stops]

        V.append(route_nodes_ordered)
        V_tamanho.append(len(route_nodes_ordered))
        route_name_map[k] = f"{key[0]} - {key[1]}"

    I = sorted(I_set, key=lambda x: (x[1], x[0]))

    # ---- Generate demand zones ----
    demand_positions, de = generate_demand_zones(
        node_positions[1:], args.num_q, rng, args.d_walk_max
    )

    # ---- Build distance matrices ----
    d_matrix = compute_walking_distances(demand_positions, node_positions[1:])
    D_list = build_route_distance_matrices(node_positions[1:], V)

    # ---- Quality scores ----
    w = generate_quality(num_n, rng)

    # ---- Sets ----
    C = build_C_set(num_n, rng)
    T = [n for n in range(1, num_n + 1) if n not in C]
    L = build_L_set(args.num_q, num_k, rng)

    # ---- Build output ----
    data = {
        "NumN": num_n,
        "NumK": num_k,
        "NumQ": args.num_q,
        "d_route_max": args.d_route_max,
        "d_walk_max": args.d_walk_max,
        "P": args.P,
        "Capt": args.capt,
        "m_max": 7,
        "omega": 0.033,
        "W1": 1,
        "W2": 1,
        "W3": 1,
        "W4": 1,
        "mu": 1,
        "theta": 1,
        "C": C,
        "T": T,
        "I": I,
        "L": L,
        "Q": list(range(1, args.num_q + 1)),
        "N": list(range(1, num_n + 1)),
        "K": list(range(1, num_k + 1)),
        "de": de,
        "w": w,
        "d": d_matrix,
        "D": D_list,
        "V": V,
        "V_tamanho": V_tamanho,
        "metadata": {
            "versao": "2.0-bh",
            "fonte": str(csv_path),
            "data_geracao": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "semente": args.seed,
            "descricao": "Dados BH convertidos do CSV da PBH",
            "num_rotas_originais": len(raw_routes),
            "rotas_selecionadas": route_name_map,
        },
        "visualizacao": {
            "posicoes_pontos": node_positions[1:],
            "posicoes_demandas": demand_positions,
            "qualidades_pontos": w,
            "niveis_demanda": de,
            "rotas": V,
        },
        "estatisticas": {
            "demanda_total": sum(de),
            "perc_acessiveis": 0,
        },
    }

    # ---- Save ----
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Nodes: {num_n}, Routes: {num_k}, Demand zones: {args.num_q}")
    print(f"  I set: {len(I)} route-stop pairs")
    print(f"  Total demand: {sum(de):.0f} passengers")
    print(f"\nRoutes:")
    for k_idx, key in enumerate(route_keys):
        print(f"  Route {k_idx+1}: {key[0]} ({key[1]}) — {V_tamanho[k_idx]} stops")

    return 0


if __name__ == "__main__":
    exit(main())
