#!/usr/bin/env python3
"""
Synthetic data generator for the SouBuz bus stop optimization model.

This module generates realistic synthetic data for testing the bus stop
optimization model, including node positions, demand zones, routes, and
all necessary parameters for the Gurobi solver.

IMPORTANT: Every candidate node belongs to at least one route.
Nodes can belong to multiple routes (intersections are allowed).

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
DEFAULT_NUM_K = 3
DEFAULT_NUM_Q = 60
DEFAULT_GRID_WIDTH = 3000.0
DEFAULT_GRID_HEIGHT = 3000.0
DEFAULT_MIN_DIST = 80.0
DEFAULT_D_WALK_MAX = 250.0
DEFAULT_D_ROUTE_MAX = 1000.0
DEFAULT_P = 1000.0
DEFAULT_CAPT = 800
DEFAULT_M_MAX = 3
DEFAULT_OMEGA = 50.0
DEFAULT_W1 = 0.35
DEFAULT_W2 = 0.15
DEFAULT_W3 = 0.30
DEFAULT_W4 = 0.20
DEFAULT_ROUTE_LEN_FRAC = 0.35
# Probability of a node belonging to multiple routes
DEFAULT_MULTI_ROUTE_PROB = 0.3


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
# FUNÇÕES DE GERAÇÃO DE ROTAS COM INTERSEÇÕES (MULTIPLAS ROTAS POR PONTO)
# ============================================================================

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
        current_pos = positions[current]
        nearest_idx = min(remaining, key=lambda i: np.linalg.norm(positions[i] - current_pos))
        route.append(nearest_idx)
        remaining.remove(nearest_idx)
        current = nearest_idx
    
    return route


def generate_routes_with_intersections(positions: np.ndarray, num_k: int, route_len: int, 
                                        multi_route_prob: float, rng) -> Tuple[List[List[int]], List[np.ndarray]]:
    """
    Generate routes where nodes can belong to multiple routes (intersections).
    
    Each node appears in at least one route, and some nodes appear in multiple
    routes to create realistic transfer points.
    
    Args:
        positions: Array of node positions (N x 2)
        num_k: Number of routes to generate
        route_len: Number of nodes per route (approximate)
        multi_route_prob: Probability of a node belonging to multiple routes
        rng: Random number generator
        
    Returns:
        routes: List of routes (each as list of node IDs, 1-indexed)
        route_indices: List of routes (each as array of indices, 0-indexed)
    """
    num_n = len(positions)
    routes = []
    route_indices = []
    
    # Build KDTree for efficient nearest neighbor queries
    tree = KDTree(positions)
    
    # Step 1: Create base assignment - each node belongs to at least one route
    # We'll assign each node to a "primary" route
    primary_route = {}
    route_nodes = {k: [] for k in range(num_k)}
    
    # Distribute nodes evenly among routes for primary assignment
    all_nodes = list(range(num_n))
    rng.shuffle(all_nodes)
    
    for i, node_idx in enumerate(all_nodes):
        route_id = i % num_k
        primary_route[node_idx] = route_id
        route_nodes[route_id].append(node_idx)
    
    # Step 2: Add secondary assignments (multi-route nodes)
    # This creates intersections between routes
    multi_route_nodes = []
    for node_idx in all_nodes:
        if rng.random() < multi_route_prob:
            # This node will belong to additional routes
            current_route = primary_route[node_idx]
            # Choose different route(s) to add this node to
            other_routes = [k for k in range(num_k) if k != current_route]
            if other_routes:
                num_extra = rng.integers(1, min(3, len(other_routes) + 1))
                extra_routes = rng.choice(other_routes, size=num_extra, replace=False)
                for extra_route in extra_routes:
                    if node_idx not in route_nodes[extra_route]:
                        route_nodes[extra_route].append(node_idx)
                        multi_route_nodes.append((node_idx, extra_route))
    
    # Step 3: Build actual routes using nearest neighbor
    for k in range(num_k):
        nodes_in_route = route_nodes[k]
        
        if len(nodes_in_route) < 2:
            # If too few nodes, add nearest neighbors
            current_set = set(nodes_in_route)
            needed = 2 - len(current_set)
            if needed > 0:
                # Find nearest neighbors to existing nodes
                for idx in list(current_set):
                    distances, neighbors = tree.query(positions[idx], k=min(10, num_n))
                    for n_idx, dist in zip(neighbors, distances):
                        if n_idx not in current_set:
                            current_set.add(n_idx)
                            needed -= 1
                            if needed <= 0:
                                break
                    if needed <= 0:
                        break
            nodes_in_route = list(current_set)
        
        # Build route using nearest neighbor
        if len(nodes_in_route) >= 2:
            route = build_route_from_points(positions, nodes_in_route, rng)
        else:
            route = nodes_in_route
        
        # Trim or pad to target length
        if len(route) > route_len:
            route = route[:route_len]
        
        # Convert to 1-indexed
        routes.append([int(n + 1) for n in route])
        route_indices.append(np.array(route))
    
    # Step 4: Ensure all nodes are covered (each node appears in at least one route)
    all_nodes_set = set(range(num_n))
    covered_nodes = set()
    for route in route_indices:
        covered_nodes.update(route)
    
    uncovered = all_nodes_set - covered_nodes
    if uncovered:
        # Add uncovered nodes to the route closest to them
        for node in uncovered:
            # Find closest existing route
            best_route = 0
            best_dist = float('inf')
            for k_idx, route in enumerate(route_indices):
                for route_node in route:
                    dist = np.linalg.norm(positions[node] - positions[route_node])
                    if dist < best_dist:
                        best_dist = dist
                        best_route = k_idx
            # Add to that route
            routes[best_route].append(node + 1)
            route_indices[best_route] = np.append(route_indices[best_route], node)
    
    # Print statistics about multi-route nodes
    node_route_count = {}
    for k, route in enumerate(route_indices):
        for node in route:
            node_route_count[node] = node_route_count.get(node, 0) + 1
    
    multi_count = sum(1 for count in node_route_count.values() if count > 1)
    print(f"  Nodes in multiple routes: {multi_count}/{num_n} ({100*multi_count/num_n:.1f}%)")
    if multi_count > 0:
        max_routes = max(node_route_count.values())
        print(f"  Max routes per node: {max_routes}")
    
    return routes, route_indices


def generate_routes_covering_all_nodes(positions: np.ndarray, num_k: int, route_len: int, rng) -> Tuple[List[List[int]], List[np.ndarray]]:
    """
    Generate routes that collectively cover all nodes (basic version).
    Each node appears in exactly one route.
    
    Args:
        positions: Array of node positions (N x 2)
        num_k: Number of routes to generate
        route_len: Number of nodes per route (approximate)
        rng: Random number generator
        
    Returns:
        routes: List of routes (each as list of node IDs, 1-indexed)
        route_indices: List of routes (each as array of indices, 0-indexed)
    """
    num_n = len(positions)
    routes = []
    route_indices = []
    
    # Build KDTree for efficient nearest neighbor queries
    tree = KDTree(positions)
    
    # Shuffle node indices for random assignment order
    all_nodes = list(range(num_n))
    rng.shuffle(all_nodes)
    
    # Distribute nodes evenly among routes
    route_nodes = {k: [] for k in range(num_k)}
    for i, node_idx in enumerate(all_nodes):
        route_id = i % num_k
        route_nodes[route_id].append(node_idx)
    
    # Build actual routes for each route using nearest neighbor
    for k in range(num_k):
        nodes_in_route = route_nodes[k]
        
        if len(nodes_in_route) < 2:
            # If too few nodes, add some from other routes
            needed = 2 - len(nodes_in_route)
            for other_node in all_nodes:
                if other_node not in nodes_in_route:
                    nodes_in_route.append(other_node)
                    needed -= 1
                    if needed <= 0:
                        break
        
        # Build route using nearest neighbor within this set
        if len(nodes_in_route) >= 2:
            route = build_route_from_points(positions, nodes_in_route, rng)
        else:
            route = nodes_in_route
        
        # Convert to 1-indexed for output
        routes.append([int(n + 1) for n in route])
        route_indices.append(np.array(route))
    
    return routes, route_indices


def generate_hybrid_routes_covering_all(positions: np.ndarray, num_k: int, route_len: int, 
                                        multi_route_prob: float, rng) -> Tuple[List[List[int]], List[np.ndarray]]:
    """
    Hybrid approach: Use K-means clustering to create geographically coherent routes
    that cover all nodes, with optional multi-route nodes.
    """
    from scipy.spatial import KDTree
    from scipy.cluster.vq import kmeans2
    
    num_n = len(positions)
    
    # Determine number of clusters (routes)
    actual_clusters = min(num_k, num_n)
    
    # Use K-means to cluster nodes
    if num_n >= actual_clusters:
        centroids, labels = kmeans2(positions, actual_clusters, minit='points', seed=rng)
    else:
        # Not enough points, simple distribution
        return generate_routes_with_intersections(positions, num_k, route_len, multi_route_prob, rng)
    
    # Group nodes by cluster (primary assignment)
    cluster_nodes = {i: [] for i in range(actual_clusters)}
    for node_idx, label in enumerate(labels):
        cluster_nodes[label].append(node_idx)
    
    # Track which nodes are in which cluster (primary)
    node_primary_cluster = {node_idx: label for node_idx, label in enumerate(labels)}
    
    # Step 2: Add secondary assignments (multi-route nodes)
    # Nodes near cluster boundaries may belong to multiple clusters
    all_nodes = list(range(num_n))
    boundary_threshold = np.percentile(positions, 70)  # Rough boundary detection
    
    for node_idx in all_nodes:
        if rng.random() < multi_route_prob:
            # Find if this node is close to other clusters
            node_pos = positions[node_idx]
            current_cluster = node_primary_cluster[node_idx]
            
            # Calculate distances to other cluster centers
            distances_to_centroids = []
            for c_idx, centroid in enumerate(centroids):
                if c_idx != current_cluster:
                    dist = np.linalg.norm(node_pos - centroid)
                    distances_to_centroids.append((c_idx, dist))
            
            distances_to_centroids.sort(key=lambda x: x[1])
            
            # Add to closest other cluster if within threshold
            for other_cluster, dist in distances_to_centroids[:2]:  # Max 2 extra routes
                if node_idx not in cluster_nodes[other_cluster]:
                    cluster_nodes[other_cluster].append(node_idx)
    
    routes = []
    route_indices = []
    
    # Ensure each cluster has at least 2 nodes
    for cluster_id in range(actual_clusters):
        if len(cluster_nodes[cluster_id]) < 2:
            current_set = set(cluster_nodes[cluster_id])
            tree = KDTree(positions)
            needed = 2 - len(current_set)
            
            for _ in range(needed):
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
            
            cluster_nodes[cluster_id] = list(current_set)
    
    # Build route for each cluster
    for cluster_id in range(actual_clusters):
        nodes_in_cluster = cluster_nodes[cluster_id]
        
        if len(nodes_in_cluster) >= 2:
            route = build_route_from_points(positions, nodes_in_cluster, rng)
        else:
            route = nodes_in_cluster
        
        # Trim to target length
        if len(route) > route_len:
            route = route[:route_len]
        
        routes.append([int(n + 1) for n in route])
        route_indices.append(np.array(route))
    
    # If we have fewer routes than requested, duplicate some routes with variations
    while len(routes) < num_k:
        template_idx = len(routes) % len(routes)
        template = routes[template_idx]
        varied = template[1:] + [template[0]]
        routes.append(varied)
        route_indices.append(np.array([int(n) - 1 for n in varied]))
    
    # Ensure all nodes are covered
    covered_nodes = set()
    for route in route_indices:
        covered_nodes.update(route)
    
    uncovered = set(range(num_n)) - covered_nodes
    if uncovered:
        # Add uncovered nodes to the closest route
        for node in uncovered:
            best_route = 0
            best_dist = float('inf')
            for k_idx, route in enumerate(route_indices):
                if len(route) > 0:
                    dist = np.linalg.norm(positions[node] - positions[route[0]])
                    if dist < best_dist:
                        best_dist = dist
                        best_route = k_idx
            routes[best_route].append(node + 1)
            route_indices[best_route] = np.append(route_indices[best_route], node)
    
    return routes, route_indices


def generate_grid_routes_covering_all(positions: np.ndarray, num_k: int, route_len: int, rng) -> Tuple[List[List[int]], List[np.ndarray]]:
    """
    Generate routes by dividing the city into vertical strips,
    ensuring all nodes are covered.
    """
    num_n = len(positions)
    
    # Sort points by X coordinate
    sorted_indices = np.argsort(positions[:, 0])
    
    # Calculate points per route
    points_per_route = max(2, num_n // num_k + 1)
    
    routes = []
    route_indices = []
    covered = set()
    
    for k in range(num_k):
        start = k * points_per_route
        end = min(start + points_per_route, num_n)
        
        if start >= num_n:
            start = start % num_n
            end = min(start + points_per_route, num_n)
        
        sector_indices = sorted_indices[start:end]
        
        if len(sector_indices) < 2:
            current_set = set(sector_indices)
            tree = KDTree(positions)
            needed = 2 - len(current_set)
            for _ in range(needed):
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
            sector_indices = list(current_set)
        
        if len(sector_indices) > 0:
            sector_positions = positions[sector_indices]
            sorted_by_y = np.argsort(sector_positions[:, 1])
            route = [sector_indices[i] for i in sorted_by_y]
            
            routes.append([int(n + 1) for n in route])
            route_indices.append(np.array(route))
            covered.update(sector_indices)
    
    # Cover any remaining uncovered nodes
    uncovered = set(range(num_n)) - covered
    if uncovered and routes:
        for node in uncovered:
            routes[0].append(node + 1)
            route_indices[0] = np.append(route_indices[0], node)
    
    return routes, route_indices


def generate_routes(positions: np.ndarray, num_k: int, route_len: int, rng, 
                   method='hybrid', multi_route_prob=DEFAULT_MULTI_ROUTE_PROB) -> Tuple[List[List[int]], List[np.ndarray]]:
    """
    Generate routes that cover all nodes.
    
    Methods:
        - 'nearest': Nearest Neighbor with coverage guarantee (no intersections by default)
        - 'grid': Divide city into vertical strips
        - 'hybrid': K-means clustering + nearest neighbor (recommended, with intersections)
    """
    if method == 'nearest':
        return generate_routes_covering_all_nodes(positions, num_k, route_len, rng)
    elif method == 'grid':
        return generate_grid_routes_covering_all(positions, num_k, route_len, rng)
    else:  # 'hybrid' (default) - with intersections
        return generate_hybrid_routes_covering_all(positions, num_k, route_len, multi_route_prob, rng)


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
    
    if not args.quiet:
        print(f"\nGenerating with {NumK} routes, {route_len} nodes per route average...")
    
    # Generate positions
    node_positions = generate_positions(NumN, grid_w, grid_h, min_dist, rng)
    demand_positions, de = generate_demand_positions(node_positions, NumQ, grid_w, grid_h, d_walk_max, rng)
    
    # Generate quality
    w = generate_quality(NumN, rng)
    
    # Generate routes (ensuring all nodes are covered, with intersections)
    routes, route_indices = generate_routes(node_positions, NumK, route_len, rng, 
                                            method=args.route_method,
                                            multi_route_prob=DEFAULT_MULTI_ROUTE_PROB)
    
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
    
    # Build output data structure
    data = {
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
        
        "C": C,
        "T": T,
        "I": I,
        "L": L,
        "Q": list(range(1, NumQ + 1)),
        "N": list(range(1, NumN + 1)),
        "K": list(range(1, NumK + 1)),
        
        "de": de,
        "w": w,
        "d": d_matrix,
        "D": D_list,
        "V": V,
        "V_tamanho": V_tamanho,
        
        "metadata": {
            "versao": "2.0",
            "data_geracao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "semente": seed,
            "descricao": "Dados sinteticos SouBuz (todos os nos pertencem a pelo menos uma rota, com interseccoes)",
        },
        
        "visualizacao": {
            "posicoes_pontos": [[round(float(x), 2), round(float(y), 2)] for x, y in node_positions],
            "posicoes_demandas": [[round(float(x), 2), round(float(y), 2)] for x, y in demand_positions],
            "qualidades_pontos": w,
            "niveis_demanda": de,
            "rotas": V,
        },
        
        "estatisticas": stats,
    }
    
    return data


# ============================================================================
# WRAPPER DE COMPATIBILIDADE (generate_single → generate_data)
# ============================================================================

def generate_single(args, seed, nprng, quiet=False):
    """Generate data (backward-compatible wrapper).
    
    Args:
        args: Command-line arguments (namespace)
        seed: Random seed
        nprng: NumPy random generator
        quiet: If True, suppress print output
    """
    args.quiet = quiet
    return generate_data(args, nprng, seed)


# ============================================================================
# INTERFACE DE LINHA DE COMANDO
# ============================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate synthetic SouBuz data")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--num-n", type=int, default=DEFAULT_NUM_N, help="Number of nodes")
    parser.add_argument("--num-k", type=int, default=DEFAULT_NUM_K, help="Number of routes")
    parser.add_argument("--num-q", type=int, default=DEFAULT_NUM_Q, help="Number of demand zones")
    parser.add_argument("--route-len", type=int, default=None, help="Nodes per route")
    parser.add_argument("--route-method", type=str, default="hybrid", 
                       choices=["nearest", "grid", "hybrid"],
                       help="Route generation method: nearest (no intersections), grid, or hybrid (with intersections)")
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
    return parser.parse_args(argv)


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
    
    # Verify coverage and count intersections
    all_nodes = set(data["N"])
    nodes_in_routes = set()
    node_route_count = {}
    
    for route in data["V"]:
        for node in route:
            nodes_in_routes.add(node)
            node_route_count[node] = node_route_count.get(node, 0) + 1
    
    missing = all_nodes - nodes_in_routes
    if missing:
        print(f"\n⚠️ WARNING: {len(missing)} nodes not in any route: {missing}")
    else:
        print(f"\n✅ All {data['NumN']} nodes are covered by at least one route!")
    
    # Show intersection statistics
    multi_route_nodes = [n for n, count in node_route_count.items() if count > 1]
    if multi_route_nodes:
        print(f"\n🔄 Intersections: {len(multi_route_nodes)} nodes belong to multiple routes")
        print(f"   Multi-route nodes: {sorted(multi_route_nodes)[:20]}{'...' if len(multi_route_nodes) > 20 else ''}")
    else:
        print(f"\n📌 No intersections (each node belongs to exactly one route)")
    
    # Print route info
    print(f"\nRoutes generated (method: {args.route_method}):")
    for k, route in enumerate(data['V']):
        print(f"  Route {k+1}: {len(route)} stops - IDs: {route[:5]}{'...' if len(route) > 5 else ''}")


if __name__ == "__main__":
    main()