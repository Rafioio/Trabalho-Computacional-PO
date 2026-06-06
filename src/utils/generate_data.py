#!/usr/bin/env python3
"""
Synthetic data generator for the SouBuz bus stop optimization model.

This module generates realistic synthetic data for testing the bus stop
optimization model, including node positions, demand zones, routes, and
all necessary parameters for the Gurobi solver.

Usage:
    python generate_data.py                              # Default: 50 nodes, 5 routes, 35 zones
    python generate_data.py --num-n 500 --num-k 10 --num-q 30
    python generate_data.py --seed 99 --output mydata.json
    python generate_data.py --scenarios 10 --prefix cenario
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set

import numpy as np
from scipy.spatial import KDTree


# ============================================================================
# CONSTANTES PADRÃO
# ============================================================================

DEFAULT_NUM_N = 200
DEFAULT_NUM_K = 2
DEFAULT_NUM_Q = 70
DEFAULT_GRID_WIDTH = 3000.0
DEFAULT_GRID_HEIGHT = 3000.0
DEFAULT_MIN_DIST = 80.0
DEFAULT_D_WALK_MAX = 400.0
DEFAULT_D_ROUTE_MAX = 800.0
DEFAULT_P = 1000.0
DEFAULT_CAPT = 800
DEFAULT_M_MAX = 7
DEFAULT_OMEGA = 0.033
DEFAULT_W1 = 1
DEFAULT_W2 = 1
DEFAULT_W3 = 1
DEFAULT_W4 = 1
DEFAULT_ROUTE_LEN_FRAC = 0.35


# ============================================================================
# FUNÇÕES DE GERAÇÃO DE POSIÇÕES (MANTIDAS)
# ============================================================================

def generate_positions(num_n, grid_w, grid_h, min_dist, rng):
    """Generate realistic node positions with clustering."""
    positions = []
    for _ in range(num_n):
        for attempt in range(2000):
            if rng.random() < 0.6 and positions:
                base = positions[rng.integers(len(positions))]
                candidate = base + rng.normal(0, [grid_w * 0.08, grid_h * 0.08])
            else:
                candidate = rng.uniform(0, [grid_w, grid_h])
            
            candidate = np.clip(candidate, 0, [grid_w, grid_h])
            
            if not positions:
                positions.append(candidate)
                break
            
            if np.all(np.linalg.norm(np.array(positions) - candidate, axis=1) >= min_dist):
                positions.append(candidate)
                break
        else:
            positions.append(rng.uniform(0, [grid_w, grid_h]))
    
    return np.array(positions)


def generate_demand_positions(positions, num_q, grid_w, grid_h, d_walk_max, rng):
    """Generate demand zones near existing nodes."""
    num_n = len(positions)
    idxs = rng.integers(num_n, size=num_q)
    use_cluster = rng.random(size=num_q) < 0.8
    angles = rng.uniform(0, 2 * np.pi, size=num_q)
    radii = rng.uniform(0, d_walk_max * 0.7, size=num_q)
    
    demand_positions = np.empty((num_q, 2))
    for i in range(num_q):
        if use_cluster[i]:
            base = positions[idxs[i]]
            demand_positions[i] = base + radii[i] * np.array([np.cos(angles[i]), np.sin(angles[i])])
        else:
            demand_positions[i] = rng.uniform(0, [grid_w, grid_h])
    
    np.clip(demand_positions, 0, [grid_w, grid_h], out=demand_positions)
    
    demand_levels = np.round(rng.lognormal(4.0, 0.8, size=num_q)).astype(int)
    np.clip(demand_levels, 5, 200, out=demand_levels)
    
    return demand_positions, demand_levels.tolist()


def generate_quality(num_n, rng):
    """Generate technical quality scores for nodes."""
    q = rng.beta(2, 2, size=num_n)
    return np.round(0.2 + 0.8 * q, 6).tolist()


# ============================================================================
# NOVA FUNÇÃO: GERAÇÃO DE ROTAS REALISTAS (VIZINHO MAIS PRÓXIMO)
# ============================================================================

def generate_realistic_routes(positions: np.ndarray, num_k: int, route_len: int, rng) -> Tuple[List[List[int]], List[np.ndarray]]:
    """
    Generate realistic routes using Nearest Neighbor algorithm.
    
    Each route starts at a random point and then repeatedly visits the
    nearest unvisited point, creating a geographically coherent path.
    
    Args:
        positions: Array of node positions (N x 2)
        num_k: Number of routes to generate
        route_len: Number of nodes per route
        rng: Random number generator
        
    Returns:
        routes: List of routes (each as list of node IDs, 1-indexed)
        route_indices: List of routes (each as array of indices, 0-indexed)
    """
    num_n = len(positions)
    routes = []
    route_indices = []
    used_nodes = set()
    
    # Build KDTree for efficient nearest neighbor queries
    tree = KDTree(positions)
    
    for k in range(num_k):
        # Find available starting points (not used in previous routes)
        available = [i for i in range(num_n) if i not in used_nodes]
        if not available:
            # If not enough available, reuse nodes (allow overlapping routes)
            available = list(range(num_n))
        
        # Random starting point
        start_idx = rng.choice(available)
        route = [start_idx]
        used_nodes.add(start_idx)
        
        # Build route using nearest neighbor
        current = start_idx
        for _ in range(route_len - 1):
            # Find neighbors of current point
            distances, indices = tree.query(positions[current], k=min(15, num_n))
            
            # Find nearest unvisited point
            next_idx = None
            for idx in indices:
                if idx not in used_nodes and idx != current:
                    next_idx = idx
                    break
            
            if next_idx is None:
                # No unvisited neighbors, pick any unvisited point
                remaining = [i for i in range(num_n) if i not in used_nodes]
                if remaining:
                    next_idx = rng.choice(remaining)
                else:
                    break
            
            route.append(next_idx)
            used_nodes.add(next_idx)
            current = next_idx
        
        # Store route (convert to 1-indexed for output)
        routes.append([int(n + 1) for n in route])
        route_indices.append(np.array(route))
    
    return routes, route_indices


def generate_grid_routes(positions: np.ndarray, num_k: int, route_len: int, rng) -> Tuple[List[List[int]], List[np.ndarray]]:
    """
    Alternative: Generate routes by dividing the city into sectors.
    Each route covers a specific geographic region.
    """
    num_n = len(positions)
    routes = []
    route_indices = []
    
    # Sort points by X coordinate to create vertical strips
    sorted_indices = np.argsort(positions[:, 0])
    
    # Divide points into num_k groups (roughly equal)
    points_per_route = max(route_len, num_n // num_k)
    
    for k in range(num_k):
        start = k * points_per_route
        end = min(start + points_per_route, num_n)
        
        if start >= num_n:
            # If not enough points, reuse from beginning with offset
            start = (k * route_len) % num_n
            end = start + route_len
        
        # Get indices for this sector
        sector_indices = sorted_indices[start:end]
        
        # Sort within sector by Y coordinate to create path
        if len(sector_indices) > 0:
            sector_positions = positions[sector_indices]
            sorted_by_y = np.argsort(sector_positions[:, 1])
            route = sector_indices[sorted_by_y].tolist()
            
            # Ensure route has correct length
            if len(route) > route_len:
                route = route[:route_len]
            
            routes.append([int(n + 1) for n in route])
            route_indices.append(np.array(route))
    
    return routes, route_indices


def generate_hybrid_routes(positions: np.ndarray, num_k: int, route_len: int, rng) -> Tuple[List[List[int]], List[np.ndarray]]:
    """
    Hybrid approach: combine Nearest Neighbor with geographic clustering.
    Creates routes that are locally coherent but cover different regions.
    """
    num_n = len(positions)
    routes = []
    route_indices = []
    
    # Use K-Means-like approach: find centroids of clusters
    from scipy.spatial import KDTree
    from scipy.cluster.vq import kmeans2
    
    # Find cluster centers (approximate regions)
    if num_n >= num_k:
        centroids, labels = kmeans2(positions, num_k, minit='points', seed=rng)
    else:
        # Not enough points, use simple division
        return generate_grid_routes(positions, num_k, route_len, rng)
    
    # For each cluster, build a route using nearest neighbor within the cluster
    for cluster_id in range(num_k):
        # Get points belonging to this cluster
        cluster_indices = np.where(labels == cluster_id)[0]
        
        if len(cluster_indices) == 0:
            continue
        
        # If cluster has too few points, expand to nearest neighbors
        if len(cluster_indices) < route_len:
            # Find nearest points from other clusters
            tree = KDTree(positions)
            needed = route_len - len(cluster_indices)
            
            # Collect all points in cluster
            current_set = set(cluster_indices)
            
            # Add nearest points iteratively
            for _ in range(needed):
                # Find point outside cluster closest to any point in cluster
                best_idx = None
                best_dist = float('inf')
                for idx in current_set:
                    distances, neighbors = tree.query(positions[idx], k=min(10, num_n))
                    for n_idx, dist in zip(neighbors, distances):
                        if n_idx not in current_set and dist < best_dist:
                            best_dist = dist
                            best_idx = n_idx
                if best_idx is not None:
                    current_set.add(best_idx)
            
            cluster_indices = list(current_set)
        
        # Build route within cluster using nearest neighbor
        if len(cluster_indices) >= 2:
            route = build_route_from_points(positions, cluster_indices, rng)
        else:
            route = cluster_indices
        
        # Trim or pad to exact length
        if len(route) > route_len:
            route = route[:route_len]
        elif len(route) < route_len:
            # Pad with nearest points
            current_set = set(route)
            tree = KDTree(positions)
            while len(route) < route_len:
                # Add nearest neighbor to any point in route
                best_idx = None
                best_dist = float('inf')
                for idx in route:
                    distances, neighbors = tree.query(positions[idx], k=min(10, num_n))
                    for n_idx, dist in zip(neighbors, distances):
                        if n_idx not in current_set and dist < best_dist:
                            best_dist = dist
                            best_idx = n_idx
                if best_idx is not None:
                    route.append(best_idx)
                    current_set.add(best_idx)
                else:
                    break
        
        routes.append([int(n + 1) for n in route])
        route_indices.append(np.array(route))
    
    return routes, route_indices


def build_route_from_points(positions: np.ndarray, indices: List[int], rng) -> List[int]:
    """Build a route from a set of points using nearest neighbor."""
    if len(indices) <= 1:
        return indices
    
    points = {i: positions[i] for i in indices}
    start_idx = rng.choice(list(points.keys()))
    route = [start_idx]
    remaining = set(points.keys()) - {start_idx}
    
    current = start_idx
    while remaining:
        # Find nearest remaining point
        current_pos = positions[current]
        nearest_idx = min(remaining, key=lambda i: np.linalg.norm(positions[i] - current_pos))
        route.append(nearest_idx)
        remaining.remove(nearest_idx)
        current = nearest_idx
    
    return route


def generate_routes(positions: np.ndarray, num_k: int, route_len: int, rng, method='hybrid') -> Tuple[List[List[int]], List[np.ndarray]]:
    """
    Generate routes using specified method.
    
    Methods:
        - 'nearest': Nearest Neighbor (creates natural paths)
        - 'grid': Divide city into vertical strips
        - 'hybrid': K-means clustering + nearest neighbor (recommended)
    """
    if method == 'nearest':
        return generate_realistic_routes(positions, num_k, route_len, rng)
    elif method == 'grid':
        return generate_grid_routes(positions, num_k, route_len, rng)
    else:  # 'hybrid' (default)
        return generate_hybrid_routes(positions, num_k, route_len, rng)


# ============================================================================
# FUNÇÕES DE CONSTRUÇÃO DE MATRIZES (MANTIDAS)
# ============================================================================

def build_distance_matrices(positions, route_indices, num_n, num_k):
    """Build distance matrices D for each route."""
    D_list = []
    for route_idxs in route_indices:
        route_pos = positions[route_idxs]
        diffs = positions[:, np.newaxis, :] - route_pos[np.newaxis, :, :]
        dists = np.round(np.linalg.norm(diffs, axis=2), 2)
        Dk = np.zeros((num_n, num_n))
        Dk[:, route_idxs] = dists
        D_list.append(Dk.tolist())
    
    return D_list


def compute_d_matrix(demand_positions, node_positions):
    """Compute Euclidean distance matrix between demands and nodes."""
    diffs = demand_positions[:, np.newaxis, :] - node_positions[np.newaxis, :, :]
    return np.round(np.linalg.norm(diffs, axis=2), 2).tolist()


def compute_statistics(d, de, w, d_walk_max):
    """Compute summary statistics."""
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


def build_I_set(routes, num_k):
    """Build I set (route-stop pairs)."""
    I = []
    for k_idx, route in enumerate(routes):
        k = k_idx + 1
        for n in route:
            I.append((n, k))
    return sorted(set(I), key=lambda x: (x[1], x[0]))


def build_L_set(num_q, num_k, rng):
    """Build L set (demand-route pairs)."""
    L = []
    for q in range(1, num_q + 1):
        k_count = max(2, min(num_k, rng.integers(2, num_k + 1)))
        selected = rng.choice(range(1, num_k + 1), size=k_count, replace=False)
        for k in selected:
            L.append((q, int(k)))
    return sorted(set(L), key=lambda x: (x[1], x[0]))


def build_C_set(num_n, rng):
    """Build C set (active centroids) - randomly selected for testing."""
    all_nodes = list(range(1, num_n + 1))
    rng.shuffle(all_nodes)
    num_c = max(1, num_n // 12)
    return sorted(all_nodes[:num_c])


# ============================================================================
# GERADOR PRINCIPAL
# ============================================================================

def generate_data(args, rng, seed):
    """Generate complete dataset."""
    
    NumN = args.num_n
    NumK = args.num_k
    NumQ = args.num_q
    route_len = args.route_len if args.route_len else max(2, int(NumN * DEFAULT_ROUTE_LEN_FRAC))
    grid_w = args.grid_width
    grid_h = args.grid_height
    min_dist = args.min_dist
    d_walk_max = args.d_walk_max
    
    # Generate positions
    node_positions = generate_positions(NumN, grid_w, grid_h, min_dist, rng)
    demand_positions, de = generate_demand_positions(node_positions, NumQ, grid_w, grid_h, d_walk_max, rng)
    
    # Generate quality
    w = generate_quality(NumN, rng)
    
    # Generate routes (using hybrid method for realistic paths)
    routes, route_indices = generate_routes(node_positions, NumK, route_len, rng, method='hybrid')
    
    # Build matrices
    d_matrix = compute_d_matrix(demand_positions, node_positions)
    D_list = build_distance_matrices(node_positions, route_indices, NumN, NumK)
    
    # Build sets
    I = build_I_set(routes, NumK)
    L = build_L_set(NumQ, NumK, rng)
    C = build_C_set(NumN, rng)
    T = [n for n in range(1, NumN + 1) if n not in C]
    
    # V (routes in order) and V_tamanho
    V = routes
    V_tamanho = [len(v) for v in V]
    
    # Compute statistics
    stats = compute_statistics(d_matrix, de, w, d_walk_max)
    
    # Build output data structure with UPPERCASE keys as expected by the model
    data = {
        # Basic parameters (UPPERCASE as expected by loader)
        "NumN": NumN,
        "NumK": NumK,
        "NumQ": NumQ,
        "d_route_max": args.d_route_max,
        "d_walk_max": d_walk_max,
        "P": args.P,
        "Capt": args.capt,
        "m_max": args.m_max,
        "omega": args.omega,
        "W1": args.W1,
        "W2": args.W2,
        "W3": args.W3,
        "W4": args.W4,
        
        # Sets (UPPERCASE)
        "C": C,
        "T": T,
        "I": I,
        "L": L,
        "Q": list(range(1, NumQ + 1)),
        "N": list(range(1, NumN + 1)),
        "K": list(range(1, NumK + 1)),
        
        # Arrays (UPPERCASE)
        "de": de,
        "w": w,
        "d": d_matrix,
        "D": D_list,
        "V": V,
        "V_tamanho": V_tamanho,
        
        # Metadata
        "metadata": {
            "versao": "2.0",
            "data_geracao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "semente": seed,
            "descricao": "Dados sinteticos SouBuz",
        },
        
        # Visualization data
        "visualizacao": {
            "posicoes_pontos": [[round(float(x), 2), round(float(y), 2)] for x, y in node_positions],
            "posicoes_demandas": [[round(float(x), 2), round(float(y), 2)] for x, y in demand_positions],
            "qualidades_pontos": w,
            "niveis_demanda": de,
            "rotas": V,
        },
        
        # Statistics
        "estatisticas": stats,
    }
    
    return data


# ============================================================================
# INTERFACE DE LINHA DE COMANDO
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic SouBuz data")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--num-n", type=int, default=DEFAULT_NUM_N, help="Number of nodes")
    parser.add_argument("--num-k", type=int, default=DEFAULT_NUM_K, help="Number of routes")
    parser.add_argument("--num-q", type=int, default=DEFAULT_NUM_Q, help="Number of demand zones")
    parser.add_argument("--route-len", type=int, default=None, help="Nodes per route")
    parser.add_argument("--route-method", type=str, default="hybrid", 
                       choices=["nearest", "grid", "hybrid"],
                       help="Route generation method: nearest, grid, or hybrid")
    parser.add_argument("--grid-width", type=float, default=DEFAULT_GRID_WIDTH, help="City width (m)")
    parser.add_argument("--grid-height", type=float, default=DEFAULT_GRID_HEIGHT, help="City height (m)")
    parser.add_argument("--min-dist", type=float, default=DEFAULT_MIN_DIST, help="Min distance between nodes (m)")
    parser.add_argument("--d-route-max", type=float, default=DEFAULT_D_ROUTE_MAX, help="Max route spacing (m)")
    parser.add_argument("--d-walk-max", type=float, default=DEFAULT_D_WALK_MAX, help="Max walk distance (m)")
    parser.add_argument("--capt", type=int, default=DEFAULT_CAPT, help="Base system capacity")
    parser.add_argument("--m-max", type=int, default=DEFAULT_M_MAX, help="Max routes per stop")
    parser.add_argument("--P", type=float, default=DEFAULT_P, help="Penalty for unserved demand")
    parser.add_argument("--omega", type=float, default=DEFAULT_OMEGA, help="Additional capacity cost weight")
    parser.add_argument("--W1", type=float, default=DEFAULT_W1, help="Weight for social cost")
    parser.add_argument("--W2", type=float, default=DEFAULT_W2, help="Weight for technical feasibility")
    parser.add_argument("--W3", type=float, default=DEFAULT_W3, help="Weight for infrastructure cost")
    parser.add_argument("--W4", type=float, default=DEFAULT_W4, help="Weight for spacing penalty")
    parser.add_argument("--output", "-o", default="dados_generated.json", help="Output JSON file")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    if args.seed is not None:
        seed = args.seed
        rng = np.random.default_rng(seed)
        print(f"[generate_data] Using fixed seed: {seed}")
    else:
        seed = int(datetime.now().timestamp() * 1_000_000) % (2**31)
        rng = np.random.default_rng(seed)
        print(f"[generate_data] Using random seed: {seed}")
    
    # Generate data
    data = generate_data(args, rng, seed)
    
    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[generate_data] Saved to {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    
    # Print summary
    stats = data["estatisticas"]
    print(f"\nSummary:")
    print(f"  Nodes: {data['NumN']}, Routes: {data['NumK']}, Demand zones: {data['NumQ']}")
    print(f"  Total demand: {stats['demanda_total']:.0f}")
    print(f"  Average quality: {stats['qualidade_media']:.3f}")
    print(f"  Accessible pairs: {stats['perc_acessiveis']:.1f}%")
    
    # Print route info
    print(f"\nRoutes generated (method: {args.route_method}):")
    for k, route in enumerate(data['V']):
        print(f"  Route {k+1}: {len(route)} stops - IDs: {route[:5]}{'...' if len(route) > 5 else ''}")


if __name__ == "__main__":
    main()