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
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

import numpy as np


# ============================================================================
# CONSTANTES PADRÃO
# ============================================================================

@dataclass(frozen=True)
class DefaultConfig:
    """Default configuration parameters for data generation."""
    
    # Dimensões
    num_n: int = 50                    # Número de nós candidatos
    num_k: int = 5                     # Número de rotas
    num_q: int = 35                    # Número de zonas de demanda
    
    # Geométricos
    grid_width: float = 3000.0         # Largura da cidade (metros)
    grid_height: float = 3000.0        # Altura da cidade (metros)
    min_distance: float = 80.0         # Distância mínima entre pontos (metros)
    
    # Parâmetros do modelo
    d_walk_max: float = 500.0          # Distância máxima de caminhada (metros)
    d_route_max: float = 800.0         # Distância máxima entre paradas (metros)
    penalty_unserved: float = 1000.0   # Penalidade por demanda não atendida (P)
    base_capacity: int = 800           # Capacidade base do sistema (Capt)
    max_routes_per_stop: int = 3       # Máximo de rotas por ponto (m_max)
    omega: float = 50.0                # Peso do custo de capacidade adicional
    
    # Pesos da função objetivo
    w1: float = 0.35                   # Peso para custo social (f1)
    w2: float = 0.15                   # Peso para viabilidade técnica (f2)
    w3: float = 0.30                   # Peso para custo infraestrutura (f3)
    w4: float = 0.20                   # Peso para penalidade espaçamento (f4)
    
    # Distribuições
    cluster_probability: float = 0.6   # Probabilidade de clusterização
    demand_near_node_prob: float = 0.8 # Probabilidade demanda perto de ponto
    route_len_fraction: float = 0.35   # Fração de nós por rota
    demand_mean: float = 4.0           # Média da log-normal para demanda
    demand_sigma: float = 0.8          # Sigma da log-normal para demanda
    demand_min: int = 5                # Demanda mínima
    demand_max: int = 200              # Demanda máxima
    quality_min: float = 0.2           # Qualidade mínima
    quality_max: float = 0.8           # Qualidade máxima (range adicional)
    
    @property
    def route_len_default(self) -> int:
        """Default number of nodes per route."""
        return max(2, int(self.num_n * self.route_len_fraction))
    
    @property
    def num_c_default(self) -> int:
        """Default number of active centroids."""
        return max(1, self.num_n // 12)


# ============================================================================
# CLASSES DE DADOS
# ============================================================================

@dataclass
class GeoData:
    """Container for geographical positions."""
    nodes: np.ndarray
    demand_zones: np.ndarray


@dataclass
class ModelData:
    """Container for all model parameters and generated data."""
    
    # Conjuntos
    num_n: int
    num_k: int
    num_q: int
    q: List[int]
    n: List[int]
    k: List[int]
    c: List[int]
    t: List[int]
    
    # Relações
    i: List[Tuple[int, int]]           # (n, k) - rota para nó
    l: List[Tuple[int, int]]           # (q, k) - demanda para rota
    
    # Parâmetros
    de: List[int]                       # Demandas
    d: List[List[float]]                # Matriz de distâncias (Q x N)
    w: List[float]                      # Qualidades dos pontos
    d_list: List[List[List[float]]]     # Matrizes D (K x N x N)
    v: List[List[int]]                  # Ordem dos pontos nas rotas
    v_tamanho: List[int]                # Tamanhos das rotas
    
    # Parâmetros escalares
    p: float
    omega: float
    w1: float
    w2: float
    w3: float
    w4: float
    capt: int
    m_max: int
    d_route_max: float
    d_walk_max: float
    
    # Metadados
    metadata: Dict[str, Any]
    visualizacao: Dict[str, Any]
    estatisticas: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert numpy arrays to lists if any remain
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
        return result


# ============================================================================
# GERADOR DE POSIÇÕES
# ============================================================================

class PositionGenerator:
    """Generates realistic positions for nodes and demand zones."""
    
    def __init__(self, config: DefaultConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
    
    def generate_nodes(self) -> np.ndarray:
        """
        Generate node positions with clustering and minimum distance constraints.
        
        Returns:
            np.ndarray: Array of shape (num_n, 2) with node coordinates
        """
        positions = []
        max_attempts = 2000
        
        for _ in range(self.config.num_n):
            for attempt in range(max_attempts):
                # Clusterização: 60% chance de nascer perto de ponto existente
                if self.rng.random() < self.config.cluster_probability and positions:
                    base = positions[self.rng.integers(len(positions))]
                    candidate = base + self.rng.normal(
                        0, 
                        [self.config.grid_width * 0.08, self.config.grid_height * 0.08]
                    )
                else:
                    candidate = self.rng.uniform(0, [self.config.grid_width, self.config.grid_height])
                
                candidate = np.clip(candidate, 0, [self.config.grid_width, self.config.grid_height])
                
                # Verificar distância mínima
                if not positions:
                    positions.append(candidate)
                    break
                
                distances = np.linalg.norm(np.array(positions) - candidate, axis=1)
                if np.all(distances >= self.config.min_distance):
                    positions.append(candidate)
                    break
            else:
                # Fallback: posição aleatória sem restrição
                positions.append(self.rng.uniform(0, [self.config.grid_width, self.config.grid_height]))
        
        return np.array(positions, dtype=np.float32)
    
    def generate_demand_zones(self, node_positions: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Generate demand zones near existing nodes.
        
        Args:
            node_positions: Array of node coordinates
            
        Returns:
            Tuple of (positions array, demand levels list)
        """
        num_nodes = len(node_positions)
        idxs = self.rng.integers(num_nodes, size=self.config.num_q)
        use_cluster = self.rng.random(size=self.config.num_q) < self.config.demand_near_node_prob
        
        angles = self.rng.uniform(0, 2 * np.pi, size=self.config.num_q)
        radii = self.rng.uniform(0, self.config.d_walk_max * 0.7, size=self.config.num_q)
        
        demand_positions = np.empty((self.config.num_q, 2))
        
        for i in range(self.config.num_q):
            if use_cluster[i]:
                base = node_positions[idxs[i]]
                demand_positions[i] = base + radii[i] * np.array([np.cos(angles[i]), np.sin(angles[i])])
            else:
                demand_positions[i] = self.rng.uniform(0, [self.config.grid_width, self.config.grid_height])
        
        np.clip(demand_positions, 0, [self.config.grid_width, self.config.grid_height], out=demand_positions)
        
        # Distribuição log-normal para demanda
        demand_levels = np.round(self.rng.lognormal(
            self.config.demand_mean, 
            self.config.demand_sigma, 
            size=self.config.num_q
        )).astype(int)
        np.clip(demand_levels, self.config.demand_min, self.config.demand_max, out=demand_levels)
        
        return demand_positions.astype(np.float32), demand_levels.tolist()


# ============================================================================
# GERADOR DE QUALIDADES
# ============================================================================

class QualityGenerator:
    """Generates technical quality scores for nodes."""
    
    def __init__(self, config: DefaultConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
    
    def generate(self) -> List[float]:
        """
        Generate quality scores using Beta distribution.
        
        Returns:
            List of quality scores in [0.2, 1.0] range
        """
        # Beta(2,2) gives symmetric distribution around 0.5
        qualities = self.rng.beta(2, 2, size=self.config.num_n)
        # Scale to [0.2, 1.0]
        qualities = self.config.quality_min + self.config.quality_max * qualities
        return np.round(qualities, 6).tolist()


# ============================================================================
# GERADOR DE ROTAS
# ============================================================================

class RouteGenerator:
    """Generates bus routes and related structures."""
    
    def __init__(self, config: DefaultConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
    
    def generate(
        self, 
        node_positions: np.ndarray
    ) -> Tuple[List[List[int]], List[int], List[Tuple[int, int]], List[List[List[float]]]]:
        """
        Generate routes and associated structures.
        
        Args:
            node_positions: Array of node coordinates
            
        Returns:
            Tuple of (V, V_tamanho, I, D_list)
        """
        num_n = self.config.num_n
        route_len = self.config.route_len_default
        
        v = []
        v_tamanho = []
        i_set = set()
        route_node_indices = []
        
        for k in range(self.config.num_k):
            # Seleciona pontos aleatórios e ordena por coordenada X
            perm = self.rng.permuted(np.arange(num_n))
            sel = perm[:route_len]
            sel_sorted = np.sort(sel[np.argsort(node_positions[sel, 0])])
            route_node_indices.append(sel_sorted)
            
            route = (sel_sorted + 1).tolist()  # 1-indexed
            v.append(route)
            v_tamanho.append(len(route))
            
            for node_id in route:
                i_set.add((node_id, k + 1))
        
        i = sorted(i_set, key=lambda x: (x[1], x[0]))
        d_list = self._build_distance_matrices(node_positions, route_node_indices)
        
        return v, v_tamanho, i, d_list
    
    def _build_distance_matrices(
        self, 
        positions: np.ndarray, 
        route_indices: List[np.ndarray]
    ) -> List[List[List[float]]]:
        """
        Build distance matrices D[k][n1][n2] for each route.
        
        Args:
            positions: Array of all node positions
            route_indices: List of node indices per route
            
        Returns:
            List of K matrices of size (N x N) with distances
        """
        num_n = positions.shape[0]
        d_list = []
        
        for route_idxs in route_indices:
            route_positions = positions[route_idxs]
            # Vectorized distance computation
            diffs = positions[:, np.newaxis, :] - route_positions[np.newaxis, :, :]
            distances = np.linalg.norm(diffs, axis=2)
            d_matrix = np.zeros((num_n, num_n))
            d_matrix[:, route_idxs] = distances
            d_list.append(np.round(d_matrix, 2).tolist())
        
        return d_list


# ============================================================================
# GERADOR DE RELAÇÕES DEMANDA-ROTA
# ============================================================================

class DemandRouteGenerator:
    """Generates L set (demand-route relationships)."""
    
    def __init__(self, config: DefaultConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
    
    def generate(self) -> List[Tuple[int, int]]:
        """
        Generate random demand-route associations.
        
        Returns:
            List of tuples (q, k) with 1-indexed IDs
        """
        l_set = set()
        
        for q in range(1, self.config.num_q + 1):
            # Cada demanda é atendida por 2 a K rotas aleatórias
            max_k = min(self.config.num_k, self.rng.integers(2, self.config.num_k + 1))
            k_count = max(2, max_k)
            selected = self.rng.choice(range(1, self.config.num_k + 1), size=k_count, replace=False)
            
            for k in selected:
                l_set.add((q, int(k)))
        
        return sorted(l_set, key=lambda x: (x[1], x[0]))


# ============================================================================
# GERADOR DE CONJUNTOS ATIVOS
# ============================================================================

class ActiveSetGenerator:
    """Generates C and T sets (active/inactive nodes)."""
    
    def __init__(self, config: DefaultConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
    
    def generate(self) -> Tuple[List[int], List[int]]:
        """
        Generate random active nodes set C.
        
        Returns:
            Tuple of (C, T) as 1-indexed lists
        """
        all_nodes = list(range(1, self.config.num_n + 1))
        self.rng.shuffle(all_nodes)
        
        num_c = self.config.num_c_default
        c = sorted(all_nodes[:num_c])
        t = sorted(all_nodes[num_c:])
        
        return c, t


# ============================================================================
# GERADOR PRINCIPAL
# ============================================================================

class SouBuzDataGenerator:
    """
    Main data generator for SouBuz optimization model.
    
    This class orchestrates all sub-generators to produce a complete
    dataset compatible with the Gurobi solver.
    """
    
    def __init__(self, config: Optional[DefaultConfig] = None, seed: Optional[int] = None):
        """
        Initialize the generator.
        
        Args:
            config: Configuration object (uses DefaultConfig if None)
            seed: Random seed (generates random seed if None)
        """
        self.config = config or DefaultConfig()
        
        # Configurar semente
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        else:
            self.seed = int(datetime.now().timestamp() * 1_000_000) % (2**31)
            self.rng = np.random.default_rng(self.seed)
        
        # Inicializar sub-generadores
        self.position_gen = PositionGenerator(self.config, self.rng)
        self.quality_gen = QualityGenerator(self.config, self.rng)
        self.route_gen = RouteGenerator(self.config, self.rng)
        self.demand_route_gen = DemandRouteGenerator(self.config, self.rng)
        self.active_set_gen = ActiveSetGenerator(self.config, self.rng)
    
    def generate(self, quiet: bool = False) -> ModelData:
        """
        Generate complete dataset.
        
        Args:
            quiet: If True, suppress console output
            
        Returns:
            ModelData object with all generated data
        """
        if not quiet:
            self._print_header()
        
        # 1. Gerar posições
        if not quiet:
            print("[1/6] Gerando posições dos pontos...")
        node_positions = self.position_gen.generate_nodes()
        demand_positions, demand_levels = self.position_gen.generate_demand_zones(node_positions)
        
        # 2. Gerar qualidades
        if not quiet:
            print("[2/6] Gerando qualidades técnicas...")
        qualities = self.quality_gen.generate()
        
        # 3. Gerar matriz de distâncias
        if not quiet:
            print("[3/6] Calculando matriz de distâncias...")
        d_matrix = self._compute_distance_matrix(demand_positions, node_positions)
        
        # 4. Gerar rotas
        if not quiet:
            print("[4/6] Gerando rotas...")
        v, v_tamanho, i_set, d_list = self.route_gen.generate(node_positions)
        
        # 5. Gerar relações demanda-rota
        if not quiet:
            print("[5/6] Gerando relações demanda-rota...")
        l_set = self.demand_route_gen.generate()
        
        # 6. Gerar conjuntos ativos
        if not quiet:
            print("[6/6] Gerando conjuntos C e T...")
        c_set, t_set = self.active_set_gen.generate()
        
        # Calcular estatísticas
        stats = self._compute_statistics(d_matrix, demand_levels, qualities)
        
        if not quiet:
            self._print_summary(stats)
        
        return self._build_model_data(
            node_positions, demand_positions, demand_levels, qualities,
            d_matrix, v, v_tamanho, i_set, l_set, c_set, t_set, d_list, stats
        )
    
    def _compute_distance_matrix(
        self, 
        demand_positions: np.ndarray, 
        node_positions: np.ndarray
    ) -> List[List[float]]:
        """Compute Euclidean distance matrix between demands and nodes."""
        diffs = demand_positions[:, np.newaxis, :] - node_positions[np.newaxis, :, :]
        distances = np.linalg.norm(diffs, axis=2)
        return np.round(distances, 2).tolist()
    
    def _compute_statistics(
        self, 
        d_matrix: List[List[float]], 
        demand_levels: List[int], 
        qualities: List[float]
    ) -> Dict[str, Any]:
        """Compute summary statistics for the generated data."""
        d_arr = np.array(d_matrix)
        accessible = d_arr[d_arr <= self.config.d_walk_max]
        de_arr = np.array(demand_levels)
        w_arr = np.array(qualities)
        
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
    
    def _build_model_data(
        self,
        node_positions: np.ndarray,
        demand_positions: np.ndarray,
        demand_levels: List[int],
        qualities: List[float],
        d_matrix: List[List[float]],
        v: List[List[int]],
        v_tamanho: List[int],
        i_set: List[Tuple[int, int]],
        l_set: List[Tuple[int, int]],
        c_set: List[int],
        t_set: List[int],
        d_list: List[List[List[float]]],
        stats: Dict[str, Any]
    ) -> ModelData:
        """Build the complete ModelData object."""
        
        return ModelData(
            num_n=self.config.num_n,
            num_k=self.config.num_k,
            num_q=self.config.num_q,
            q=list(range(1, self.config.num_q + 1)),
            n=list(range(1, self.config.num_n + 1)),
            k=list(range(1, self.config.num_k + 1)),
            c=c_set,
            t=t_set,
            i=i_set,
            l=l_set,
            de=demand_levels,
            d=d_matrix,
            w=qualities,
            d_list=d_list,
            v=v,
            v_tamanho=v_tamanho,
            p=self.config.penalty_unserved,
            omega=self.config.omega,
            w1=self.config.w1,
            w2=self.config.w2,
            w3=self.config.w3,
            w4=self.config.w4,
            capt=self.config.base_capacity,
            m_max=self.config.max_routes_per_stop,
            d_route_max=self.config.d_route_max,
            d_walk_max=self.config.d_walk_max,
            metadata={
                "versao": "2.0",
                "data_geracao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "semente": self.seed,
                "descricao": "Dados sinteticos SouBuz (src/utils/generate_data.py)",
            },
            visualizacao={
                "posicoes_pontos": [[round(float(x), 2), round(float(y), 2)] for x, y in node_positions],
                "posicoes_demandas": [[round(float(x), 2), round(float(y), 2)] for x, y in demand_positions],
                "qualidades_pontos": qualities,
                "niveis_demanda": demand_levels,
            },
            estatisticas=stats,
        )
    
    def _print_header(self) -> None:
        """Print generation header."""
        print("=" * 60)
        print("GERADOR DE DADOS SINTÉTICOS - SouBuz")
        print("=" * 60)
        print(f"\nConfiguração: {self.config.num_n} nós, {self.config.num_k} rotas, {self.config.num_q} zonas")
        print(f"Semente: {self.seed}\n")
    
    def _print_summary(self, stats: Dict[str, Any]) -> None:
        """Print summary statistics."""
        print(f"\n📊 Estatísticas geradas:")
        print(f"  - Demanda total: {stats['demanda_total']} passageiros")
        print(f"  - Demanda média: {stats['demanda_media']:.1f}")
        print(f"  - Pontos acessíveis: {stats['perc_acessiveis']:.1f}%")
        print(f"  - Qualidade média dos pontos: {stats['qualidade_media']:.3f}")


# ============================================================================
# FUNÇÕES DE EXPORTAÇÃO
# ============================================================================

def export_to_json(data: ModelData, filename: str, quiet: bool = False) -> None:
    """
    Export generated data to JSON file.
    
    Args:
        data: ModelData object to export
        filename: Output file path
        quiet: If True, suppress console output
    """
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data.to_dict(), f, indent=2, ensure_ascii=False)
    
    if not quiet:
        size_kb = output_path.stat().st_size / 1024
        print(f"\n💾 Arquivo '{filename}' criado ({size_kb:.1f} KB)")


def generate_scenarios(
    num_scenarios: int, 
    prefix: str = "cenario",
    base_config: Optional[DefaultConfig] = None,
    quiet: bool = False
) -> List[Dict[str, Any]]:
    """
    Generate multiple scenarios in batch mode.
    
    Args:
        num_scenarios: Number of scenarios to generate
        prefix: Prefix for output filenames
        base_config: Base configuration (uses DefaultConfig if None)
        quiet: If True, suppress console output
        
    Returns:
        List of scenario metadata
    """
    scenarios = []
    
    for i in range(1, num_scenarios + 1):
        # Cada cenário usa sua própria semente aleatória
        scenario_seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        generator = SouBuzDataGenerator(base_config, seed=scenario_seed)
        data = generator.generate(quiet=quiet)
        
        filename = f"{prefix}_{i:02d}.json"
        export_to_json(data, filename, quiet=quiet)
        
        scenarios.append({
            "cenario": i,
            "arquivo": filename,
            "semente": scenario_seed,
            "demanda_total": data.estatisticas["demanda_total"],
        })
        
        if not quiet:
            print(f"  [{i}/{num_scenarios}] {filename}")
    
    # Salvar resumo
    if scenarios:
        resumo_path = Path(prefix).parent / "resumo_cenarios.json"
        with open(resumo_path, "w", encoding="utf-8") as f:
            json.dump(scenarios, f, indent=2, ensure_ascii=False)
        
        if not quiet:
            print(f"\n📋 Resumo salvo em {resumo_path}")
    
    return scenarios


# ============================================================================
# INTERFACE DE LINHA DE COMANDO
# ============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic SouBuz data for bus stop optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python generate_data.py                              # Configuração padrão
  python generate_data.py --num-n 500 --num-k 10       # 500 nós, 10 rotas
  python generate_data.py --seed 42 --output dados.json
  python generate_data.py --scenarios 10 --prefix teste
        """
    )
    
    # Dimensões
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--num-n", type=int, default=DefaultConfig.num_n, help="Number of nodes")
    parser.add_argument("--num-k", type=int, default=DefaultConfig.num_k, help="Number of routes")
    parser.add_argument("--num-q", type=int, default=DefaultConfig.num_q, help="Number of demand zones")
    parser.add_argument("--num-c", type=int, default=None, help="Number of centroids (default: ~8% of N)")
    parser.add_argument("--route-len", type=int, default=None, help="Nodes per route (default: ~35% of N)")
    
    # Geométricos
    parser.add_argument("--grid-width", type=float, default=DefaultConfig.grid_width, help="City width (m)")
    parser.add_argument("--grid-height", type=float, default=DefaultConfig.grid_height, help="City height (m)")
    parser.add_argument("--min-dist", type=float, default=DefaultConfig.min_distance, help="Min distance between nodes (m)")
    
    # Parâmetros do modelo
    parser.add_argument("--d-route-max", type=float, default=DefaultConfig.d_route_max, help="Max route spacing (m)")
    parser.add_argument("--d-walk-max", type=float, default=DefaultConfig.d_walk_max, help="Max walk distance (m)")
    parser.add_argument("--capt", type=int, default=DefaultConfig.base_capacity, help="Base system capacity")
    parser.add_argument("--m-max", type=int, default=DefaultConfig.max_routes_per_stop, help="Max routes per stop")
    parser.add_argument("--P", type=float, default=DefaultConfig.penalty_unserved, help="Penalty for unserved demand")
    parser.add_argument("--omega", type=float, default=DefaultConfig.omega, help="Additional capacity cost weight")
    
    # Pesos da FO
    parser.add_argument("--W1", type=float, default=DefaultConfig.w1, help="Weight for social cost")
    parser.add_argument("--W2", type=float, default=DefaultConfig.w2, help="Weight for technical feasibility")
    parser.add_argument("--W3", type=float, default=DefaultConfig.w3, help="Weight for infrastructure cost")
    parser.add_argument("--W4", type=float, default=DefaultConfig.w4, help="Weight for spacing penalty")
    
    # Saída
    parser.add_argument("--output", default=None, help="Output JSON file")
    parser.add_argument("--scenarios", type=int, default=None, help="Batch-generate N scenarios")
    parser.add_argument("--prefix", default="cenario", help="Prefix for batch output files")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    
    return parser.parse_args(argv)


def build_config_from_args(args: argparse.Namespace) -> DefaultConfig:
    """Build DefaultConfig from parsed arguments."""
    # Create a dict with only the attributes that exist in DefaultConfig
    config_dict = {
        'num_n': args.num_n,
        'num_k': args.num_k,
        'num_q': args.num_q,
        'grid_width': args.grid_width,
        'grid_height': args.grid_height,
        'min_distance': args.min_dist,
        'd_walk_max': args.d_walk_max,
        'd_route_max': args.d_route_max,
        'penalty_unserved': args.P,
        'base_capacity': args.capt,
        'max_routes_per_stop': args.m_max,
        'omega': args.omega,
        'w1': args.W1,
        'w2': args.W2,
        'w3': args.W3,
        'w4': args.W4,
    }
    
    # Add custom values if provided
    if args.num_c is not None:
        config_dict['num_c_default'] = args.num_c
    if args.route_len is not None:
        config_dict['route_len_fraction'] = args.route_len / args.num_n
    
    return DefaultConfig(**config_dict)


# ============================================================================
# PONTO DE ENTRADA PRINCIPAL
# ============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point."""
    args = parse_args(argv)
    
    # Configurar
    config = build_config_from_args(args)
    
    # Modo de múltiplos cenários
    if args.scenarios:
        generate_scenarios(
            num_scenarios=args.scenarios,
            prefix=args.prefix,
            base_config=config,
            quiet=args.quiet
        )
        return
    
    # Modo de cenário único
    generator = SouBuzDataGenerator(config, seed=args.seed)
    data = generator.generate(quiet=args.quiet)
    
    output = args.output or "dados_generated.json"
    export_to_json(data, output, quiet=args.quiet)


if __name__ == "__main__":
    main()