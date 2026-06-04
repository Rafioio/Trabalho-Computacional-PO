#!/usr/bin/env python3
"""
Map viewer for SouBuz - Visualizes input data and optimization results.
- Input mode: shows candidate stops and demand zones (no routes)
- Solution mode: shows active stops and routes from solution
- Includes: accessibility circle, density analysis, comparative view

Usage:
    python map_viewer.py dados_generated.json                    # Input data only
    python map_viewer.py dados_generated.json --solution solucao.json  # With solution
    python map_viewer.py --density-only dados_generated.json     # Density analysis only
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import numpy as np


# ============================================================================
# CONFIGURAÇÃO DE ESTILO
# ============================================================================

class StyleConfig:
    """Visualization style configuration."""
    
    # Colors
    COLOR_DEMAND = "#87CEEB"           # Light blue
    COLOR_DEMAND_EDGE = "#1E90FF"      # Dark blue
    COLOR_ACTIVE_STOP = "#2E8B57"      # Sea green
    COLOR_ACTIVE_STOP_EDGE = "#1a5a38" # Dark green
    COLOR_CANDIDATE_STOP = "#D3D3D3"   # Light gray
    COLOR_CANDIDATE_EDGE = "#A9A9A9"   # Dark gray
    COLOR_BACKGROUND = "#F8F9FA"       # Light gray background
    COLOR_INFO_BOX = "#FFFFE0"         # Light yellow
    COLOR_GRID = "#E0E0E0"             # Light gray for grid
    COLOR_ACCESSIBILITY = "#1E90FF"    # Blue for accessibility circle
    
    # Route colors (cycle through these)
    ROUTE_COLORS = [
        "#FF6B6B",  # Red
        "#4ECDC4",  # Teal
        "#45B7D1",  # Blue
        "#96CEB4",  # Green
        "#FFEAA7",  # Yellow
        "#DDA0DD",  # Purple
        "#98D8C8",  # Mint
        "#F7B731",  # Orange
        "#A8E6CF",  # Light green
        "#FF8B94",  # Pink
    ]
    
    # Font sizes
    FONTSIZE_TITLE = 14
    FONTSIZE_LABEL = 11
    FONTSIZE_ANNOTATION = 8
    FONTSIZE_LEGEND = 9
    FONTSIZE_INFO = 9
    
    # Sizes
    MARKER_SIZE_ACTIVE = 250
    MARKER_SIZE_CANDIDATE = 150
    MARKER_SIZE_DEMAND_BASE = 80
    LINE_WIDTH_ROUTE = 2.5
    ACCESSIBILITY_ALPHA = 0.15
    
    # Padding
    PADDING = 0.12


# ============================================================================
# CARREGAMENTO DE DADOS
# ============================================================================

def load_json(filepath: str) -> Optional[Dict[str, Any]]:
    """Load JSON data from file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None


def load_data(filepath: str) -> Optional[Dict[str, Any]]:
    """Load input data (dados_generated.json format)."""
    raw = load_json(filepath)
    if raw is None:
        return None
    
    # Check if it's the new format (with 'parametros')
    if "parametros" in raw:
        params = raw["parametros"]
        data = {
            "NumN": params.get("NumN", 0),
            "NumK": params.get("NumK", 0),
            "NumQ": params.get("NumQ", 0),
            "d_walk_max": params.get("d_walk_max", 500),
            "d_route_max": params.get("d_route_max", 800),
            "P": params.get("P", 1000),
            "Capt": params.get("Capt", 800),
            "m_max": params.get("m_max", 3),
            "omega": params.get("omega", 50),
            "W1": params.get("W1", 0.35),
            "W2": params.get("W2", 0.15),
            "W3": params.get("W3", 0.30),
            "W4": params.get("W4", 0.20),
            "C": raw.get("C", []),
            "I": raw.get("I", []),
            "L": raw.get("L", []),
            "de": raw.get("de", []),
            "w": raw.get("w", []),
            "d": raw.get("d", []),
            "D": raw.get("D", []),
            "V": raw.get("V", []),
            "V_tamanho": raw.get("V_tamanho", []),
            "visualizacao": raw.get("visualizacao", {}),
            "estatisticas": raw.get("estatisticas", {}),
        }
        # Fix missing fields
        data["N"] = list(range(1, data["NumN"] + 1))
        data["K"] = list(range(1, data["NumK"] + 1))
        data["Q"] = list(range(1, data["NumQ"] + 1))
        data["T"] = [n for n in data["N"] if n not in data["C"]]
    else:
        # Direct format
        data = raw
        if "N" not in data:
            data["N"] = list(range(1, data.get("NumN", 0) + 1))
        if "K" not in data:
            data["K"] = list(range(1, data.get("NumK", 0) + 1))
        if "Q" not in data:
            data["Q"] = list(range(1, data.get("NumQ", 0) + 1))
        if "T" not in data and "C" in data:
            data["T"] = [n for n in data["N"] if n not in data["C"]]
    
    return data


def load_solution(filepath: str) -> Optional[Dict[str, Any]]:
    """Load solution file (solucao.json format)."""
    return load_json(filepath)


# ============================================================================
# FUNÇÕES AUXILIARES DE VISUALIZAÇÃO
# ============================================================================

def compute_limits(positions: List[List[float]], padding: float = 0.12) -> Tuple[float, float, float, float]:
    """Compute plot limits with padding."""
    if not positions:
        return (0, 10, 0, 10)
    
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    
    dx = (xmax - xmin) or 1
    dy = (ymax - ymin) or 1
    
    return (xmin - dx * padding, xmax + dx * padding,
            ymin - dy * padding, ymax + dy * padding)


def draw_grid(ax, xmin: float, xmax: float, ymin: float, ymax: float, style: StyleConfig):
    """Draw adaptive grid lines."""
    x_range = xmax - xmin
    y_range = ymax - ymin
    
    nx = max(5, min(10, int(x_range / 300)))
    ny = max(5, min(10, int(y_range / 300)))
    
    x_ticks = np.linspace(xmin, xmax, nx + 1)
    y_ticks = np.linspace(ymin, ymax, ny + 1)
    
    for x in x_ticks:
        ax.axvline(x, color=style.COLOR_GRID, ls="--", lw=0.5, alpha=0.5, zorder=0)
    for y in y_ticks:
        ax.axhline(y, color=style.COLOR_GRID, ls="--", lw=0.5, alpha=0.5, zorder=0)


def draw_scale_bar(ax, xmin: float, xmax: float, ymin: float, ymax: float, style: StyleConfig):
    """Draw scale bar at bottom-right corner."""
    width = xmax - xmin
    height = ymax - ymin
    
    # Calculate nice scale length
    target = width * 0.15
    for unit in [500, 200, 100, 50, 25, 10, 5]:
        if target >= unit * 1.5:
            length = round(target / unit) * unit
            break
    else:
        length = round(target)
    
    x_start = xmax - width * 0.12
    x_end = x_start + length
    y_scale = ymin + height * 0.05
    
    ax.plot([x_start, x_end], [y_scale, y_scale], "k-", lw=3, zorder=10)
    tick_h = height * 0.008
    ax.plot([x_start, x_start], [y_scale - tick_h, y_scale + tick_h], "k-", lw=2, zorder=10)
    ax.plot([x_end, x_end], [y_scale - tick_h, y_scale + tick_h], "k-", lw=2, zorder=10)
    
    ax.text((x_start + x_end) / 2, y_scale - height * 0.015,
            f"{int(length)} m", ha="center", va="top",
            fontsize=style.FONTSIZE_LEGEND, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))


def draw_compass(ax, xmin: float, xmax: float, ymin: float, ymax: float, style: StyleConfig):
    """Draw compass rose at top-right corner."""
    width = xmax - xmin
    height = ymax - ymin
    
    cx = xmax - width * 0.08
    cy = ymax - height * 0.08
    r = min(width, height) * 0.04
    
    ax.add_patch(Circle((cx, cy), r, fill=False, ec="#333", lw=1.5, zorder=10))
    ax.plot([cx, cx], [cy - r, cy + r], "#333", lw=1.2, zorder=10)
    ax.plot([cx - r, cx + r], [cy, cy], "#333", lw=1.2, zorder=10)
    
    ax.annotate("N", (cx, cy + r), xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333")
    ax.annotate("S", (cx, cy - r), xytext=(0, -3), textcoords="offset points",
                ha="center", va="top", fontsize=8, color="#333")
    ax.annotate("L", (cx + r, cy), xytext=(3, 0), textcoords="offset points",
                ha="left", va="center", fontsize=8, color="#333")
    ax.annotate("O", (cx - r, cy), xytext=(-3, 0), textcoords="offset points",
                ha="right", va="center", fontsize=8, color="#333")


def draw_accessibility_circle(ax, positions_demand, d_walk_max, style: StyleConfig):
    """Draw accessibility circle for a sample demand node."""
    if not positions_demand:
        return
    
    # Use the first demand node as example
    cx, cy = positions_demand[0]
    circle = Circle((cx, cy), d_walk_max, fill=False, 
                    edgecolor=style.COLOR_ACCESSIBILITY, linestyle="--", 
                    linewidth=1.5, alpha=0.5, zorder=1)
    ax.add_patch(circle)
    
    ax.text(cx + d_walk_max * 0.6, cy + d_walk_max * 0.6,
            f"Raio de acessibilidade\n{d_walk_max:.0f} m",
            fontsize=style.FONTSIZE_ANNOTATION, alpha=0.7, ha="center",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))


def draw_routes_solution(ax, data: Dict[str, Any], solution: Dict[str, Any],
                         positions: List[List[float]], style: StyleConfig):
    """
    Draw routes as solid lines connecting stops based on solution.
    """
    V = data.get("V", [])
    if not V:
        return
    
    # Get active route-stops from solution
    active_route_stops = set()
    x_k = solution.get("x_k", {})
    for key, val in x_k.items():
        if val > 0.5:
            if isinstance(key, str) and key.startswith("("):
                parts = key.strip("()").split(",")
                if len(parts) == 2:
                    active_route_stops.add((int(parts[0]), int(parts[1])))
            elif isinstance(key, (list, tuple)) and len(key) == 2:
                active_route_stops.add((key[0], key[1]))
    
    # Also check x_k_ativos if present (alternative format)
    x_k_ativos = solution.get("x_k_ativos", [])
    for item in x_k_ativos:
        active_route_stops.add((item["n"], item["k"]))
    
    # Draw each route
    for k_idx, route_nodes in enumerate(V):
        color = style.ROUTE_COLORS[k_idx % len(style.ROUTE_COLORS)]
        
        # Get coordinates
        coords = []
        for node_id in route_nodes:
            if 1 <= node_id <= len(positions):
                coords.append(positions[node_id - 1])
        
        if len(coords) >= 2:
            xs, ys = zip(*coords)
            ax.plot(xs, ys, color=color, linewidth=style.LINE_WIDTH_ROUTE,
                   alpha=0.8, zorder=2, solid_capstyle='round')


# ============================================================================
# VISUALIZAÇÃO PRINCIPAL - DADOS DE ENTRADA
# ============================================================================

def create_input_figure(data: Dict[str, Any], style: StyleConfig = None) -> plt.Figure:
    """Create figure for input data only (NO routes)."""
    if style is None:
        style = StyleConfig()
    
    viz = data.get("visualizacao", {})
    positions_nodes = viz.get("posicoes_pontos", [])
    positions_demand = viz.get("posicoes_demandas", [])
    qualities = viz.get("qualidades_pontos", [])
    demand_levels = viz.get("niveis_demanda", [])
    d_walk_max = data.get("d_walk_max", 500)
    
    all_positions = positions_nodes + positions_demand
    xmin, xmax, ymin, ymax = compute_limits(all_positions, style.PADDING)
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_facecolor(style.COLOR_BACKGROUND)
    ax.set_xlabel("Coordenada X (m)", fontsize=style.FONTSIZE_LABEL, fontweight="bold")
    ax.set_ylabel("Coordenada Y (m)", fontsize=style.FONTSIZE_LABEL, fontweight="bold")
    
    draw_grid(ax, xmin, xmax, ymin, ymax, style)
    
    # Draw accessibility circle (example)
    draw_accessibility_circle(ax, positions_demand, d_walk_max, style)
    
    # Draw demand zones
    max_demand = max(demand_levels) if demand_levels else 1
    for i, (x, y) in enumerate(positions_demand):
        size = style.MARKER_SIZE_DEMAND_BASE + (demand_levels[i] / max_demand) * style.MARKER_SIZE_DEMAND_BASE
        ax.scatter(x, y, s=size, c=style.COLOR_DEMAND, alpha=0.7,
                  edgecolors=style.COLOR_DEMAND_EDGE, linewidth=1.5, zorder=3)
        if len(positions_demand) <= 30:
            ax.annotate(f"D{i+1}", (x, y), xytext=(5, 5), textcoords="offset points",
                       fontsize=style.FONTSIZE_ANNOTATION,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    
    # Draw candidate stops (all stops are candidates in input)
    for i, (x, y) in enumerate(positions_nodes):
        quality = qualities[i] if i < len(qualities) else 0.5
        # Color based on quality (red=bad, green=good)
        color = (max(0.2, 1 - quality), max(0.3, quality), 0.2)
        ax.scatter(x, y, s=style.MARKER_SIZE_CANDIDATE, c=[color],
                  marker="o", edgecolors=style.COLOR_CANDIDATE_EDGE,
                  linewidth=2, alpha=0.85, zorder=4)
        ax.annotate(str(i + 1), (x, y), ha="center", va="center",
                   fontsize=9, fontweight="bold", color="white", zorder=5)
    
    draw_scale_bar(ax, xmin, xmax, ymin, ymax, style)
    draw_compass(ax, xmin, xmax, ymin, ymax, style)
    
    # Title and info
    stats = data.get("estatisticas", {})
    title = f"DADOS DE ENTRADA (PRÉ-OTIMIZAÇÃO)\n{len(positions_nodes)} Pontos Candidatos | {len(positions_demand)} Nós de Demanda"
    ax.set_title(title, fontsize=style.FONTSIZE_TITLE, fontweight="bold", pad=20)
    
    info_text = (f"PARÂMETROS DO MODELO\n"
                 f"d_walk_max = {d_walk_max} m\n"
                 f"d_route_max = {data.get('d_route_max', 800)} m\n"
                 f"Capt = {data.get('Capt', 800)}\n"
                 f"m_max = {data.get('m_max', 3)} rotas/ponto\n"
                 f"Demanda total = {stats.get('demanda_total', 0)}")
    
    ax.text(xmin, ymax, info_text, transform=ax.transData,
            fontsize=style.FONTSIZE_INFO, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor=style.COLOR_INFO_BOX, alpha=0.9))
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Nó de Demanda",
               markerfacecolor=style.COLOR_DEMAND, markersize=10,
               markeredgecolor=style.COLOR_DEMAND_EDGE, markeredgewidth=1.5),
        Line2D([0], [0], marker="o", color="w", label="Ponto Candidato (cor = qualidade)",
               markerfacecolor=(0.6, 0.6, 0.2), markersize=10, markeredgecolor="#333"),
        Line2D([0], [0], linestyle="--", color=style.COLOR_ACCESSIBILITY, linewidth=1.5,
               label=f"Raio de Acessibilidade ({d_walk_max}m)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left",
             fontsize=style.FONTSIZE_LEGEND, framealpha=0.9)
    
    fig.tight_layout()
    return fig


# ============================================================================
# VISUALIZAÇÃO PRINCIPAL - SOLUÇÃO
# ============================================================================

def create_solution_figure(data: Dict[str, Any], solution: Dict[str, Any],
                           style: StyleConfig = None) -> plt.Figure:
    """Create figure for solution (active stops and routes)."""
    if style is None:
        style = StyleConfig()
    
    viz = data.get("visualizacao", {})
    positions_nodes = viz.get("posicoes_pontos", [])
    positions_demand = viz.get("posicoes_demandas", [])
    demand_levels = viz.get("niveis_demanda", [])
    d_walk_max = data.get("d_walk_max", 500)
    
    # Get active stops from solution
    active_stops = set(solution.get("pontos_ativos", []))
    
    all_positions = positions_nodes + positions_demand
    xmin, xmax, ymin, ymax = compute_limits(all_positions, style.PADDING)
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_facecolor(style.COLOR_BACKGROUND)
    ax.set_xlabel("Coordenada X (m)", fontsize=style.FONTSIZE_LABEL, fontweight="bold")
    ax.set_ylabel("Coordenada Y (m)", fontsize=style.FONTSIZE_LABEL, fontweight="bold")
    
    draw_grid(ax, xmin, xmax, ymin, ymax, style)
    
    # Draw accessibility circle (example)
    draw_accessibility_circle(ax, positions_demand, d_walk_max, style)
    
    # Draw ROUTES (solid lines)
    draw_routes_solution(ax, data, solution, positions_nodes, style)
    
    # Draw demand zones
    max_demand = max(demand_levels) if demand_levels else 1
    for i, (x, y) in enumerate(positions_demand):
        size = style.MARKER_SIZE_DEMAND_BASE + (demand_levels[i] / max_demand) * style.MARKER_SIZE_DEMAND_BASE
        ax.scatter(x, y, s=size, c=style.COLOR_DEMAND, alpha=0.7,
                  edgecolors=style.COLOR_DEMAND_EDGE, linewidth=1.5, zorder=3)
        if len(positions_demand) <= 30:
            ax.annotate(f"D{i+1}", (x, y), xytext=(5, 5), textcoords="offset points",
                       fontsize=style.FONTSIZE_ANNOTATION,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    
    # Draw active stops (green squares) and inactive stops (gray circles)
    for i, (x, y) in enumerate(positions_nodes):
        node_id = i + 1
        is_active = node_id in active_stops
        
        if is_active:
            ax.scatter(x, y, s=style.MARKER_SIZE_ACTIVE, c=[style.COLOR_ACTIVE_STOP],
                      marker="s", edgecolors=style.COLOR_ACTIVE_STOP_EDGE,
                      linewidth=2.5, alpha=0.95, zorder=4)
            ax.annotate(str(node_id), (x, y), ha="center", va="center",
                       fontsize=10, fontweight="bold", color="white", zorder=5)
        else:
            ax.scatter(x, y, s=style.MARKER_SIZE_CANDIDATE * 0.6, c=["lightgray"],
                      marker="o", edgecolors="gray", linewidth=1, alpha=0.5, zorder=3)
            ax.annotate(str(node_id), (x, y), ha="center", va="center",
                       fontsize=7, color="gray", alpha=0.7, zorder=4)
    
    draw_scale_bar(ax, xmin, xmax, ymin, ymax, style)
    draw_compass(ax, xmin, xmax, ymin, ymax, style)
    
    # Title
    n_active = len(active_stops)
    n_total = len(positions_nodes)
    title = f"SOLUÇÃO DO MODELO DE OTIMIZAÇÃO\n{n_active} Pontos Ativos (de {n_total}) | {len(positions_demand)} Demandas"
    title += f"\nValor Objetivo: {solution.get('valor_objetivo', 0):.2f}"
    ax.set_title(title, fontsize=style.FONTSIZE_TITLE, fontweight="bold", pad=20)
    
    # Info box
    info_text = (f"RESULTADOS\n"
                 f"f₁ (custo social): {solution.get('f1', 0):.2f}\n"
                 f"f₂ (viabilidade técnica): {solution.get('f2', 0):.2f}\n"
                 f"f₃ (custo infraestrutura): {solution.get('f3', 0):.2f}\n"
                 f"f₄ (penalidade espaçamento): {solution.get('f4', 0):.2f}\n"
                 f"Cad (capacidade adicional): {solution.get('Cad', 0):.0f}")
    
    ax.text(xmin, ymax, info_text, transform=ax.transData,
            fontsize=style.FONTSIZE_INFO, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor=style.COLOR_INFO_BOX, alpha=0.9))
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Nó de Demanda",
               markerfacecolor=style.COLOR_DEMAND, markersize=10,
               markeredgecolor=style.COLOR_DEMAND_EDGE, markeredgewidth=1.5),
        Line2D([0], [0], marker="s", color="w", label="Ponto Ativo (selecionado)",
               markerfacecolor=style.COLOR_ACTIVE_STOP, markersize=10,
               markeredgecolor=style.COLOR_ACTIVE_STOP_EDGE),
        Line2D([0], [0], marker="o", color="w", label="Ponto Inativo",
               markerfacecolor="lightgray", markersize=8, markeredgecolor="gray"),
        Line2D([0], [0], color="gray", linewidth=2, label="Rota (linha cheia)"),
        Line2D([0], [0], linestyle="--", color=style.COLOR_ACCESSIBILITY, linewidth=1.5,
               label=f"Raio de Acessibilidade ({d_walk_max}m)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left",
             fontsize=style.FONTSIZE_LEGEND, framealpha=0.9)
    
    fig.tight_layout()
    return fig


# ============================================================================
# ANÁLISE DE DENSIDADE
# ============================================================================

def create_density_figure(data: Dict[str, Any], style: StyleConfig = None) -> plt.Figure:
    """Create density analysis visualizations (hexbin + histograms)."""
    if style is None:
        style = StyleConfig()
    
    viz = data.get("visualizacao", {})
    positions_nodes = viz.get("posicoes_pontos", [])
    positions_demand = viz.get("posicoes_demandas", [])
    qualities = viz.get("qualidades_pontos", [])
    demand_levels = viz.get("niveis_demanda", [])
    stats = data.get("estatisticas", {})
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Density of candidate stops (hexbin)
    ax1 = axes[0, 0]
    if positions_nodes:
        xs = [p[0] for p in positions_nodes]
        ys = [p[1] for p in positions_nodes]
        hb = ax1.hexbin(xs, ys, gridsize=25, cmap="YlOrRd", alpha=0.7, mincnt=1)
        ax1.scatter(xs, ys, c="blue", s=20, alpha=0.5, edgecolors="black", linewidth=0.5)
        plt.colorbar(hb, ax=ax1, label="Densidade")
    ax1.set_title("Densidade dos Pontos Candidatos", fontsize=12, fontweight="bold")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    
    # 2. Density of demand zones (hexbin)
    ax2 = axes[0, 1]
    if positions_demand:
        xs = [p[0] for p in positions_demand]
        ys = [p[1] for p in positions_demand]
        hb = ax2.hexbin(xs, ys, gridsize=25, cmap="Blues", alpha=0.7, mincnt=1)
        ax2.scatter(xs, ys, c="red", s=20, alpha=0.5, edgecolors="black", linewidth=0.5)
        plt.colorbar(hb, ax=ax2, label="Densidade")
    ax2.set_title("Densidade dos Nós de Demanda", fontsize=12, fontweight="bold")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    
    # 3. Distribution of quality scores (histogram)
    ax3 = axes[1, 0]
    if qualities:
        ax3.hist(qualities, bins=20, color="green", alpha=0.7, edgecolor="black")
        ax3.axvline(stats.get("qualidade_media", 0), color="red", linestyle="--",
                   label=f"Média: {stats.get('qualidade_media', 0):.3f}")
        ax3.legend()
    ax3.set_title("Distribuição da Qualidade dos Pontos (w[n])", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Qualidade")
    ax3.set_ylabel("Frequência")
    
    # 4. Distribution of demand levels (histogram)
    ax4 = axes[1, 1]
    if demand_levels:
        ax4.hist(demand_levels, bins=20, color="blue", alpha=0.7, edgecolor="black")
        ax4.axvline(stats.get("demanda_media", 0), color="red", linestyle="--",
                   label=f"Média: {stats.get('demanda_media', 0):.1f}")
        ax4.legend()
    ax4.set_title("Distribuição dos Níveis de Demanda (de[q])", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Demanda (passageiros)")
    ax4.set_ylabel("Frequência")
    
    seed = data.get("metadata", {}).get("semente", "?")
    fig.suptitle(f"Análise Estatística dos Dados (semente: {seed})",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    
    return fig


# ============================================================================
# COMPARAÇÃO ANTES/DEPOIS
# ============================================================================

def create_comparison_figure(data: Dict[str, Any], solution: Dict[str, Any],
                              style: StyleConfig = None) -> plt.Figure:
    """Create before/after comparison figure."""
    if style is None:
        style = StyleConfig()
    
    viz = data.get("visualizacao", {})
    positions_nodes = viz.get("posicoes_pontos", [])
    positions_demand = viz.get("posicoes_demandas", [])
    qualities = viz.get("qualidades_pontos", [])
    demand_levels = viz.get("niveis_demanda", [])
    
    active_stops = set(solution.get("pontos_ativos", []))
    V = data.get("V", [])
    
    all_positions = positions_nodes + positions_demand
    xmin, xmax, ymin, ymax = compute_limits(all_positions, style.PADDING)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # ========== BEFORE ==========
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax1.set_facecolor(style.COLOR_BACKGROUND)
    ax1.set_xlabel("X (m)", fontsize=style.FONTSIZE_LABEL)
    ax1.set_ylabel("Y (m)", fontsize=style.FONTSIZE_LABEL)
    draw_grid(ax1, xmin, xmax, ymin, ymax, style)
    
    # Demand zones
    max_demand = max(demand_levels) if demand_levels else 1
    for i, (x, y) in enumerate(positions_demand):
        size = style.MARKER_SIZE_DEMAND_BASE + (demand_levels[i] / max_demand) * style.MARKER_SIZE_DEMAND_BASE
        ax1.scatter(x, y, s=size, c=style.COLOR_DEMAND, alpha=0.7,
                   edgecolors=style.COLOR_DEMAND_EDGE, linewidth=1.5, zorder=3)
    
    # Candidate stops
    for i, (x, y) in enumerate(positions_nodes):
        quality = qualities[i] if i < len(qualities) else 0.5
        color = (max(0.2, 1 - quality), max(0.3, quality), 0.2)
        ax1.scatter(x, y, s=style.MARKER_SIZE_CANDIDATE, c=[color],
                   marker="o", edgecolors=style.COLOR_CANDIDATE_EDGE,
                   linewidth=2, alpha=0.85, zorder=4)
        ax1.annotate(str(i + 1), (x, y), ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white", zorder=5)
    
    ax1.set_title(f"ANTES: {len(positions_nodes)} Pontos Candidatos",
                  fontsize=style.FONTSIZE_TITLE, fontweight="bold")
    
    # ========== AFTER ==========
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.set_facecolor(style.COLOR_BACKGROUND)
    ax2.set_xlabel("X (m)", fontsize=style.FONTSIZE_LABEL)
    ax2.set_ylabel("Y (m)", fontsize=style.FONTSIZE_LABEL)
    draw_grid(ax2, xmin, xmax, ymin, ymax, style)
    
    # Routes
    draw_routes_solution(ax2, data, solution, positions_nodes, style)
    
    # Demand zones
    for i, (x, y) in enumerate(positions_demand):
        size = style.MARKER_SIZE_DEMAND_BASE + (demand_levels[i] / max_demand) * style.MARKER_SIZE_DEMAND_BASE
        ax2.scatter(x, y, s=size, c=style.COLOR_DEMAND, alpha=0.7,
                   edgecolors=style.COLOR_DEMAND_EDGE, linewidth=1.5, zorder=3)
    
    # Active and inactive stops
    for i, (x, y) in enumerate(positions_nodes):
        node_id = i + 1
        is_active = node_id in active_stops
        
        if is_active:
            ax2.scatter(x, y, s=style.MARKER_SIZE_ACTIVE, c=[style.COLOR_ACTIVE_STOP],
                       marker="s", edgecolors=style.COLOR_ACTIVE_STOP_EDGE,
                       linewidth=2.5, alpha=0.95, zorder=4)
            ax2.annotate(str(node_id), (x, y), ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white", zorder=5)
        else:
            ax2.scatter(x, y, s=style.MARKER_SIZE_CANDIDATE * 0.6, c=["lightgray"],
                       marker="o", edgecolors="gray", linewidth=1, alpha=0.5, zorder=3)
            ax2.annotate(str(node_id), (x, y), ha="center", va="center",
                        fontsize=7, color="gray", alpha=0.7, zorder=4)
    
    ax2.set_title(f"DEPOIS: {len(active_stops)} Pontos Ativos\nValor Objetivo: {solution.get('valor_objetivo', 0):.2f}",
                  fontsize=style.FONTSIZE_TITLE, fontweight="bold")
    
    # Scale bars and compass
    for ax in [ax1, ax2]:
        draw_scale_bar(ax, xmin, xmax, ymin, ymax, style)
        draw_compass(ax, xmin, xmax, ymin, ymax, style)
    
    fig.suptitle("COMPARAÇÃO ANTES/DEPOIS DA OTIMIZAÇÃO",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    
    return fig


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SouBuz Map Viewer")
    parser.add_argument("data_file", nargs="?", default=None,
                       help="Path to input data JSON file (dados_generated.json)")
    parser.add_argument("--solution", "-s", default=None,
                       help="Path to solution JSON file (solucao.json)")
    parser.add_argument("--density-only", action="store_true",
                       help="Show only density analysis")
    parser.add_argument("--comparison", action="store_true",
                       help="Show before/after comparison (requires solution)")
    parser.add_argument("--output", "-o", default=None,
                       help="Output image file (e.g., figure.png)")
    args = parser.parse_args()
    
    style = StyleConfig()
    
    # Load data
    data = None
    if args.data_file:
        data = load_data(args.data_file)
        if data is None:
            sys.exit(1)
        print(f"Loaded input data: {data.get('NumN', 0)} nodes, {data.get('NumK', 0)} routes, {data.get('NumQ', 0)} demand zones")
    
    # Load solution
    solution = None
    if args.solution:
        solution = load_solution(args.solution)
        if solution:
            print(f"Loaded solution: {len(solution.get('pontos_ativos', []))} active stops, objective={solution.get('valor_objetivo', 0):.2f}")
    
    # Create figures based on options
    figures = []
    
    if args.density_only and data:
        print("\nGenerating density analysis...")
        fig = create_density_figure(data, style)
        figures.append(("densidade.png" if not args.output else None, fig))
        plt.show()
    
    elif args.comparison and solution and data:
        print("\nGenerating comparison figure...")
        fig = create_comparison_figure(data, solution, style)
        figures.append(("comparacao.png" if not args.output else None, fig))
        plt.show()
    
    elif solution and data:
        print("\nGenerating solution figure...")
        fig = create_solution_figure(data, solution, style)
        figures.append(("solucao.png" if not args.output else None, fig))
        plt.show()
    
    elif data:
        print("\nGenerating input figure...")
        fig = create_input_figure(data, style)
        figures.append(("entrada.png" if not args.output else None, fig))
        
        # Also show density analysis
        print("\nGenerating density analysis...")
        fig2 = create_density_figure(data, style)
        figures.append(("densidade.png" if not args.output else None, fig2))
        plt.show()
    
    else:
        print("Error: Need at least data_file")
        print("Usage: python map_viewer.py dados_generated.json [--solution solucao.json]")
        sys.exit(1)
    
    # Save figures if output specified
    if args.output and figures:
        for name, fig in figures:
            if name:
                output_path = args.output
            else:
                output_path = args.output
            fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"Figure saved to: {output_path}")
            break  # Save only the first figure if single output specified


if __name__ == "__main__":
    main()