#!/usr/bin/env python3
"""
Map viewer and density analyzer for SouBuz scenario data.

Reads the flat JSON format produced by src/utils/generate_data.py and
produces publication-quality visualizations: city map with demand zones,
candidate stops (colored by quality), accessibility circles, scale bar,
compass rose, density hexbin plots, and before/after comparison when a
solution JSON is present.

Usage:
    python src/scripts/map_viewer.py                     # loads dados_generated.json
    python src/scripts/map_viewer.py cenario_01.json     # specific file
    python src/scripts/map_viewer.py --output figures/   # save to custom directory
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.axes import Axes


# ============================================================================
# CONSTANTES
# ============================================================================

@dataclass(frozen=True)
class StyleConfig:
    """Visualization style configuration."""
    
    # Cores
    color_demand: str = "#87CEEB"           # Azul claro para demandas
    color_demand_edge: str = "#1E90FF"      # Azul forte para bordas
    color_active_stop: str = "#2E8B57"      # Verde marinho para pontos ativos
    color_active_stop_edge: str = "#1a5a38" # Verde escuro para bordas
    color_inactive_stop_edge: str = "#333333"  # Cinza escuro
    color_grid: str = "lightgray"           # Cinza claro para grid
    color_background: str = "#f8f9fa"       # Fundo claro
    color_info_box: str = "#FFFFE0"         # Amarelo claro para caixa de info
    color_compass: str = "#333333"          # Cinza escuro para rosa dos ventos
    
    # Fontes
    fontsize_title: int = 14
    fontsize_label: int = 11
    fontsize_annotation: int = 8
    fontsize_legend: int = 9
    fontsize_info: int = 9
    fontsize_scale: int = 10
    
    # Geometria
    grid_padding: float = 0.12
    scale_bar_width: float = 0.12           # % da largura do gráfico
    scale_bar_height: float = 0.05          # % da altura do gráfico
    compass_margin: float = 0.08            # % da largura/altura
    compass_size: float = 0.04              # % do menor lado
    
    # Tamanhos base
    base_stop_size: float = 80              # Tamanho mínimo do ponto
    max_stop_size: float = 300              # Tamanho máximo do ponto
    base_demand_size: float = 30            # Tamanho mínimo da demanda
    max_demand_size: float = 150            # Tamanho máximo da demanda
    
    # Qualidade
    quality_min: float = 0.2
    quality_max: float = 0.8


# ============================================================================
# CLASSE PRINCIPAL DO VISUALIZADOR
# ============================================================================

class SouBuzViewer:
    """
    Map viewer and density analyzer for SouBuz scenario data.
    
    This class handles loading, processing, and visualizing data from the
    SouBuz optimization model, including city maps, density plots, and
    comparative visualizations before/after optimization.
    """
    
    def __init__(self, style: Optional[StyleConfig] = None):
        """
        Initialize the viewer with optional style configuration.
        
        Args:
            style: Style configuration (uses default if None)
        """
        self.style = style or StyleConfig()
    
    @staticmethod
    def load_data(filepath: str) -> Optional[Dict[str, Any]]:
        """
        Load JSON data from file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Dictionary with loaded data or None if file not found
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Erro: arquivo '{filepath}' nao encontrado.")
            return None
    
    @staticmethod
    def load_solution(filepath: str = "solucao.json") -> Optional[Dict[str, Any]]:
        """
        Load solution JSON if available.
        
        Args:
            filepath: Path to solution JSON file
            
        Returns:
            Dictionary with solution data or None if not found
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None
    
    def compute_limits(
        self, 
        positions: List[List[float]], 
        padding: Optional[float] = None
    ) -> Tuple[float, float, float, float]:
        """
        Compute plot limits with adaptive padding.
        
        Args:
            positions: List of [x, y] coordinates
            padding: Padding factor (uses style default if None)
            
        Returns:
            Tuple of (xmin, xmax, ymin, ymax)
        """
        if not positions:
            return (0, 10, 0, 10)
        
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        
        dx = (xmax - xmin) or 1
        dy = (ymax - ymin) or 1
        
        pad = padding if padding is not None else self.style.grid_padding
        
        return (xmin - dx * pad, xmax + dx * pad, 
                ymin - dy * pad, ymax + dy * pad)
    
    def draw_grid(self, ax: Axes, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
        """
        Draw adaptive grid lines.
        
        Args:
            ax: Matplotlib axes
            xmin, xmax: X-axis limits
            ymin, ymax: Y-axis limits
        """
        x_range = xmax - xmin
        y_range = ymax - ymin
        
        # Número adaptativo de divisões (entre 5 e 10)
        nx = max(5, min(10, int(x_range / 300)))
        ny = max(5, min(10, int(y_range / 300)))
        
        # Linhas verticais
        x_ticks = [xmin + i * x_range / nx for i in range(nx + 1)]
        for x in x_ticks:
            ax.axvline(x, color=self.style.color_grid, ls="--", 
                      lw=0.5, alpha=0.5, zorder=0)
        
        # Linhas horizontais
        y_ticks = [ymin + i * y_range / ny for i in range(ny + 1)]
        for y in y_ticks:
            ax.axhline(y, color=self.style.color_grid, ls="--", 
                      lw=0.5, alpha=0.5, zorder=0)
    
    def draw_scale_bar(self, ax: Axes, xmin: float, xmax: float, 
                       ymin: float, ymax: float) -> None:
        """
        Draw scale bar at bottom-right corner.
        
        Args:
            ax: Matplotlib axes
            xmin, xmax, ymin, ymax: Plot limits
        """
        width = xmax - xmin
        height = ymax - ymin
        
        # Calcular comprimento da escala
        target_length = width * self.style.scale_bar_width
        length = self._round_to_nice_number(target_length)
        
        # Posições
        x_start = xmax - width * self.style.scale_bar_width
        x_end = x_start + length
        y_scale = ymin + height * self.style.scale_bar_height
        
        # Linha principal
        ax.plot([x_start, x_end], [y_scale, y_scale], "k-", 
                lw=3, zorder=10)
        
        # Extremidades
        tick_height = height * 0.008
        ax.plot([x_start, x_start], [y_scale - tick_height, y_scale + tick_height], 
                "k-", lw=2, zorder=10)
        ax.plot([x_end, x_end], [y_scale - tick_height, y_scale + tick_height], 
                "k-", lw=2, zorder=10)
        
        # Texto
        ax.text((x_start + x_end) / 2, y_scale - height * 0.015, 
                f"{int(length)} m", ha="center", va="top", 
                fontsize=self.style.fontsize_scale, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", 
                         alpha=0.9, edgecolor="gray"))
    
    def draw_compass(self, ax: Axes, xmin: float, xmax: float, 
                     ymin: float, ymax: float) -> None:
        """
        Draw compass rose at top-right corner.
        
        Args:
            ax: Matplotlib axes
            xmin, xmax, ymin, ymax: Plot limits
        """
        width = xmax - xmin
        height = ymax - ymin
        
        cx = xmax - width * self.style.compass_margin
        cy = ymax - height * self.style.compass_margin
        radius = min(width, height) * self.style.compass_size
        
        # Círculo externo
        ax.add_patch(Circle((cx, cy), radius, fill=False, 
                           ec=self.style.color_compass, lw=1.5, zorder=10))
        
        # Linhas cruzadas
        ax.plot([cx, cx], [cy - radius, cy + radius], self.style.color_compass, 
                lw=1.2, zorder=10)
        ax.plot([cx - radius, cx + radius], [cy, cy], self.style.color_compass, 
                lw=1.2, zorder=10)
        
        # Rótulos
        ax.annotate("N", (cx, cy + radius), textcoords="offset points", 
                   xytext=(0, 3), ha="center", va="bottom", 
                   fontsize=9, fontweight="bold", color=self.style.color_compass)
        
        for label, dx, dy, ha, va in [
            ("S", 0, -3, "center", "top"),
            ("L", 3, 0, "left", "center"),
            ("O", -3, 0, "right", "center")
        ]:
            ax.annotate(label, (cx + (radius if label == "L" else -radius if label == "O" else cx), 
                               cy + (0 if label in ("L", "O") else -radius if label == "S" else radius)),
                       textcoords="offset points", xytext=(dx, dy),
                       ha=ha, va=va, fontsize=8, color=self.style.color_compass)
    
    @staticmethod
    def _round_to_nice_number(value: float) -> float:
        """
        Round a number to a "nice" value for scale bars.
        
        Args:
            value: Number to round
            
        Returns:
            Rounded number
        """
        for unit in [500, 200, 100, 50, 25, 10, 5]:
            if value >= unit * 1.5:
                return round(value / unit) * unit
        return round(value)
    
    def _compute_sizes(self, x_range: float) -> Tuple[float, float]:
        """
        Compute base sizes for stops and demand zones.
        
        Args:
            x_range: X-axis range (width of plot)
            
        Returns:
            Tuple of (stop_size, demand_size)
        """
        stop_size = max(self.style.base_stop_size, 
                       min(self.style.max_stop_size, x_range * 0.03))
        demand_size = max(self.style.base_demand_size, 
                         min(self.style.max_demand_size, x_range * 0.02))
        return stop_size, demand_size
    
    def _get_stop_color(self, quality: float, is_active: bool) -> Tuple[float, float, float]:
        """
        Get color for a stop based on quality and activation status.
        
        Args:
            quality: Quality score (0-1)
            is_active: Whether the stop is active in solution
            
        Returns:
            RGB tuple
        """
        if is_active:
            return (float(int(self.style.color_active_stop[1:3], 16) / 255),
                   float(int(self.style.color_active_stop[3:5], 16) / 255),
                   float(int(self.style.color_active_stop[5:7], 16) / 255))
        else:
            # Gradiente: vermelho (baixa qualidade) -> verde (alta qualidade)
            return (max(self.style.quality_min, 1 - quality),
                    max(self.style.quality_min, quality),
                    self.style.quality_min)
    
    def create_map_figure(
        self, 
        data: Dict[str, Any], 
        solution: Optional[Dict[str, Any]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Create main map visualization.
        
        Args:
            data: Scenario data dictionary
            solution: Optional solution dictionary
            
        Returns:
            Tuple of (figure, axes)
        """
        vis = data.get("visualizacao", {})
        stats = data.get("estatisticas", {})
        d_walk_max = data.get("d_walk_max", 500)
        
        positions_nodes = vis.get("posicoes_pontos", [])
        positions_demand = vis.get("posicoes_demandas", [])
        qualities = vis.get("qualidades_pontos", [])
        demand_levels = vis.get("niveis_demanda", [])
        
        has_solution = solution is not None
        active_stops = set(solution.get("pontos_ativos", [])) if has_solution else set()
        
        # Calcular limites
        all_positions = positions_nodes + positions_demand
        xmin, xmax, ymin, ymax = self.compute_limits(all_positions)
        
        # Criar figura com proporção adequada
        aspect_ratio = (xmax - xmin) / ((ymax - ymin) or 1)
        fig_width = 14
        fig_height = fig_width / aspect_ratio if aspect_ratio > 0 else 10
        
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax.set_xlabel("Coordenada X (m)", fontsize=self.style.fontsize_label, fontweight="bold")
        ax.set_ylabel("Coordenada Y (m)", fontsize=self.style.fontsize_label, fontweight="bold")
        ax.set_facecolor(self.style.color_background)
        
        # Grid
        self.draw_grid(ax, xmin, xmax, ymin, ymax)
        
        # Tamanhos base
        x_range = xmax - xmin
        stop_size, demand_size = self._compute_sizes(x_range)
        max_demand = max(demand_levels) if demand_levels else 1
        
        # Demandas
        for i, (x, y) in enumerate(positions_demand):
            size = demand_size + (demand_levels[i] / max_demand) * demand_size
            ax.scatter(x, y, s=size, c=self.style.color_demand, alpha=0.7,
                      ec=self.style.color_demand_edge, lw=1.5, zorder=2)
            
            if len(positions_demand) <= 30:
                ax.annotate(f"D{i+1}", (x, y), textcoords="offset points", 
                           xytext=(5, 5), fontsize=self.style.fontsize_annotation,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7), zorder=3)
        
        # Pontos candidatos
        for i, (x, y) in enumerate(positions_nodes):
            is_active = (i + 1) in active_stops
            quality = qualities[i] if i < len(qualities) else 0.5
            size = stop_size + quality * stop_size * 0.5
            color = self._get_stop_color(quality, is_active and has_solution)
            marker = "s" if (is_active and has_solution) else "o"
            
            ax.scatter(x, y, s=size, c=[color], marker=marker,
                      ec=self.style.color_active_stop_edge if is_active and has_solution 
                         else self.style.color_inactive_stop_edge,
                      lw=2, zorder=4, alpha=0.95 if is_active else 0.85)
            
            ax.annotate(str(i + 1), (x, y), ha="center", va="center",
                       fontsize=9, fontweight="bold", color="white", zorder=5)
        
        # Círculo de acessibilidade (exemplo)
        if positions_demand:
            cx, cy = positions_demand[0]
            circ = Circle((cx, cy), d_walk_max, fill=False, 
                         ec=self.style.color_demand_edge, ls="--", 
                         lw=1.5, alpha=0.4, zorder=1)
            ax.add_patch(circ)
            ax.text(cx + d_walk_max * 0.6, cy + d_walk_max * 0.6,
                   f"Raio de acessibilidade\n{d_walk_max:.0f} m",
                   fontsize=self.style.fontsize_annotation, alpha=0.6, ha="center",
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
        
        # Escala e rosa dos ventos
        self.draw_scale_bar(ax, xmin, xmax, ymin, ymax)
        self.draw_compass(ax, xmin, xmax, ymin, ymax)
        
        # Título
        n_nodes = len(positions_nodes)
        n_active = len(active_stops)
        n_demand = len(positions_demand)
        
        if has_solution:
            title = f"SOLUCAO DO MODELO DE OTIMIZACAO\n{n_nodes} Pontos | {n_active} Ativos | {n_demand} Demandas"
            title += f"\nValor Objetivo: {solution.get('valor_objetivo', 0):.2f}"
        else:
            title = f"DADOS DE ENTRADA (PRE-OTIMIZACAO)\n{n_nodes} Pontos Candidatos | {n_demand} Nos de Demanda"
        
        ax.set_title(title, fontsize=self.style.fontsize_title, fontweight="bold", pad=20)
        
        # Caixa de informações
        info_text = (
            f"PARAMETROS DO MODELO:\n"
            f"• d_walk_max = {d_walk_max:.0f} m\n"
            f"• d_route_max = {data.get('d_route_max', 800):.0f} m\n"
            f"• Capt = {data.get('Capt', 800)}\n"
            f"• m_max = {data.get('m_max', 3)} rotas/ponto\n"
            f"• ω = {data.get('omega', 50)}\n"
            f"• Demanda total = {stats.get('demanda_total', 0)}"
        )
        ax.text(xmin, ymax, info_text, transform=ax.transData, 
               fontsize=self.style.fontsize_info,
               verticalalignment="top", horizontalalignment="left",
               bbox=dict(boxstyle="round", facecolor=self.style.color_info_box, 
                        alpha=0.9, edgecolor="#CCC"), zorder=10)
        
        # Legenda
        if has_solution:
            legend_elements = [
                Line2D([0], [0], marker="o", color="w", mfc=self.style.color_demand, ms=10,
                      label="Nos de Demanda", mec=self.style.color_demand_edge, mew=1.5),
                Line2D([0], [0], marker="o", color="w", mfc="gray", ms=10,
                      label="Ponto (nao ativo)", mec="#333"),
                Line2D([0], [0], marker="s", color="w", mfc=self.style.color_active_stop, ms=10,
                      label="Ponto Ativo (selecionado)", mec=self.style.color_active_stop_edge),
                Line2D([0], [0], ls="--", color=self.style.color_demand_edge, lw=1.5, alpha=0.6,
                      label=f"Raio de Acessibilidade ({d_walk_max:.0f}m)"),
            ]
        else:
            legend_elements = [
                Line2D([0], [0], marker="o", color="w", mfc=self.style.color_demand, ms=10,
                      label="Nos de Demanda", mec=self.style.color_demand_edge, mew=1.5),
                Line2D([0], [0], marker="o", color="w", mfc=(0.6, 0.6, 0.2), ms=10,
                      label="Ponto Candidato (cor = qualidade)", mec="#333"),
                Line2D([0], [0], ls="--", color=self.style.color_demand_edge, lw=1.5, alpha=0.6,
                      label=f"Raio de Acessibilidade ({d_walk_max:.0f}m)"),
            ]
        
        ax.legend(handles=legend_elements, loc="lower left", 
                 fontsize=self.style.fontsize_legend, framealpha=0.9, edgecolor="#CCC")
        
        fig.tight_layout()
        return fig, ax
    
    def create_density_figure(self, data: Dict[str, Any]) -> Tuple[Figure, Any]:
        """
        Create density analysis visualizations.
        
        Args:
            data: Scenario data dictionary
            
        Returns:
            Tuple of (figure, axes)
        """
        vis = data.get("visualizacao", {})
        stats = data.get("estatisticas", {})
        
        positions_nodes = vis.get("posicoes_pontos", [])
        positions_demand = vis.get("posicoes_demandas", [])
        qualities = vis.get("qualidades_pontos", [])
        demand_levels = vis.get("niveis_demanda", [])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Densidade dos pontos
        ax = axes[0, 0]
        if positions_nodes:
            xs = [p[0] for p in positions_nodes]
            ys = [p[1] for p in positions_nodes]
            hb = ax.hexbin(xs, ys, gridsize=25, cmap="YlOrRd", alpha=0.7, mincnt=1)
            ax.scatter(xs, ys, c="blue", s=20, alpha=0.5)
            plt.colorbar(hb, ax=ax, label="Densidade")
        ax.set_title("Densidade dos Pontos Candidatos", fontsize=12, fontweight="bold")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        
        # 2. Densidade das demandas
        ax = axes[0, 1]
        if positions_demand:
            xs = [p[0] for p in positions_demand]
            ys = [p[1] for p in positions_demand]
            hb = ax.hexbin(xs, ys, gridsize=25, cmap="Blues", alpha=0.7, mincnt=1)
            ax.scatter(xs, ys, c="red", s=20, alpha=0.5)
            plt.colorbar(hb, ax=ax, label="Densidade")
        ax.set_title("Densidade dos Nos de Demanda", fontsize=12, fontweight="bold")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        
        # 3. Distribuição das qualidades
        ax = axes[1, 0]
        if qualities:
            ax.hist(qualities, bins=20, color="green", alpha=0.7, edgecolor="black")
            ax.axvline(stats.get("qualidade_media", 0), color="red", ls="--",
                      label=f"Media: {stats.get('qualidade_media', 0):.3f}")
            ax.legend()
        ax.set_title("Distribuicao da Qualidade dos Pontos (w[n])", fontsize=12, fontweight="bold")
        ax.set_xlabel("Qualidade")
        ax.set_ylabel("Frequencia")
        
        # 4. Distribuição das demandas
        ax = axes[1, 1]
        if demand_levels:
            ax.hist(demand_levels, bins=20, color="blue", alpha=0.7, edgecolor="black")
            ax.axvline(stats.get("demanda_media", 0), color="red", ls="--",
                      label=f"Media: {stats.get('demanda_media', 0):.1f}")
            ax.legend()
        ax.set_title("Distribuicao dos Niveis de Demanda (de[q])", fontsize=12, fontweight="bold")
        ax.set_xlabel("Demanda (passageiros)")
        ax.set_ylabel("Frequencia")
        
        seed = data.get("metadata", {}).get("semente", "?")
        fig.suptitle(f"Analise Estatistica dos Dados (semente: {seed})",
                    fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        
        return fig, axes
    
    def create_comparison_figure(
        self, 
        data: Dict[str, Any], 
        solution: Dict[str, Any]
    ) -> Tuple[Figure, Tuple[Axes, Axes]]:
        """
        Create before/after comparison visualization.
        
        Args:
            data: Scenario data dictionary
            solution: Solution dictionary
            
        Returns:
            Tuple of (figure, axes tuple)
        """
        vis = data.get("visualizacao", {})
        positions_nodes = vis.get("posicoes_pontos", [])
        qualities = vis.get("qualidades_pontos", [])
        active_stops = set(solution.get("pontos_ativos", []))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Antes (todos candidatos)
        for i, (x, y) in enumerate(positions_nodes):
            q = qualities[i] if i < len(qualities) else 0.5
            c = (max(0.2, 1 - q), max(0.3, q), 0.2)
            ax1.scatter(x, y, s=200, c=[c], marker="o", ec="#333", lw=2, alpha=0.8)
            ax1.annotate(str(i + 1), (x, y), ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")
        ax1.set_title(f"ANTES: {len(positions_nodes)} Pontos Candidatos", 
                     fontsize=12, fontweight="bold")
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        
        # Depois (apenas ativos)
        for i, (x, y) in enumerate(positions_nodes):
            is_active = (i + 1) in active_stops
            if is_active:
                ax2.scatter(x, y, s=250, c=[self.style.color_active_stop], 
                           marker="s", ec=self.style.color_active_stop_edge, 
                           lw=2.5, alpha=0.95)
                ax2.annotate(str(i + 1), (x, y), ha="center", va="center",
                            fontsize=9, fontweight="bold", color="white")
            else:
                ax2.scatter(x, y, s=100, c=["lightgray"], marker="o", 
                           ec="gray", lw=1, alpha=0.5)
                ax2.annotate(str(i + 1), (x, y), ha="center", va="center",
                            fontsize=7, color="gray")
        ax2.set_title(f"DEPOIS: {len(active_stops)} Pontos Ativos Selecionados",
                     fontsize=12, fontweight="bold")
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        
        # Mesmos limites
        xmin, xmax, ymin, ymax = self.compute_limits(positions_nodes, padding=0.1)
        ax1.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax2.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        
        fig.suptitle(f"Comparacao Antes/Depois da Otimizacao\n"
                    f"Valor Objetivo: {solution.get('valor_objetivo', 0):.2f}",
                    fontsize=14, fontweight="bold")
        fig.tight_layout()
        
        return fig, (ax1, ax2)
    
    def save_figures(
        self, 
        figures: List[Tuple[Figure, str]], 
        output_dir: Union[str, Path] = "."
    ) -> None:
        """
        Save figures to disk.
        
        Args:
            figures: List of (figure, filename) tuples
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for fig, filename in figures:
            filepath = output_path / filename
            fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"  - {filepath}")


# ============================================================================
# FUNÇÕES DE INTERFACE (COMPATIBILIDADE RETROATIVA)
# ============================================================================

def ler_dados_json(path="dados_generated.json"):
    """Compatibility wrapper for load_data."""
    return SouBuzViewer.load_data(path)


def ler_solucao_json(path="solucao.json"):
    """Compatibility wrapper for load_solution."""
    return SouBuzViewer.load_solution(path)


def visualizar_pontos(dados, solucao=None):
    """Compatibility wrapper for create_map_figure."""
    viewer = SouBuzViewer()
    return viewer.create_map_figure(dados, solucao)


def visualizar_densidade(dados):
    """Compatibility wrapper for create_density_figure."""
    viewer = SouBuzViewer()
    return viewer.create_density_figure(dados)


def visualizar_comparativo(dados, solucao):
    """Compatibility wrapper for create_comparison_figure."""
    viewer = SouBuzViewer()
    return viewer.create_comparison_figure(dados, solucao)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Map viewer and density analyzer for SouBuz scenario data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input_file", 
        nargs="?", 
        default="dados_generated.json",
        help="Input JSON file (default: dados_generated.json)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=".",
        help="Output directory for saved figures (default: current directory)"
    )
    parser.add_argument(
        "--solution", "-s",
        default="solucao.json",
        help="Solution JSON file (default: solucao.json)"
    )
    parser.add_argument(
        "--no-save", 
        action="store_true",
        help="Don't prompt to save figures"
    )
    return parser.parse_args(argv)


# ============================================================================
# PONTO DE ENTRADA PRINCIPAL
# ============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point."""
    args = parse_args(argv)
    
    print("=" * 60)
    print("VISUALIZADOR DO SISTEMA DE PARADAS DE ONIBUS")
    print("=" * 60)
    
    print(f"\nCarregando '{args.input_file}'...")
    data = ler_dados_json(args.input_file)
    if data is None:
        return
    
    solution = ler_solucao_json(args.solution) if not args.no_save else None
    
    # Exibir resumo
    print(f"\nDados carregados:")
    print(f"  - Pontos candidatos: {data.get('NumN', '?')}")
    print(f"  - Rotas: {data.get('NumK', '?')}")
    print(f"  - Nos de demanda: {data.get('NumQ', '?')}")
    stats = data.get("estatisticas", {})
    print(f"  - Demanda total: {stats.get('demanda_total', '?')} passageiros")
    print(f"  - Qualidade media dos pontos: {stats.get('qualidade_media', '?'):.3f}")
    print(f"  - d_walk_max: {data.get('d_walk_max', 500)} m")
    
    if solution:
        print(f"\nSolucao encontrada:")
        print(f"  - Valor objetivo: {solution.get('valor_objetivo', 0):.2f}")
        print(f"  - Pontos ativos: {len(solution.get('pontos_ativos', []))}")
    
    # Criar visualizações
    viewer = SouBuzViewer()
    
    print("\nGerando visualizacao principal...")
    fig1, _ = viewer.create_map_figure(data, solution)
    
    print("Gerando analise de densidade...")
    fig2, _ = viewer.create_density_figure(data)
    
    figures_to_save = [(fig1, "visualizacao_pontos.png"), 
                      (fig2, "visualizacao_densidade.png")]
    
    if solution:
        print("Gerando visualizacao comparativa...")
        fig3, _ = viewer.create_comparison_figure(data, solution)
        figures_to_save.append((fig3, "visualizacao_comparativo.png"))
    
    plt.show()
    
    # Salvar figuras
    if not args.no_save:
        try:
            resp = input("\nDeseja salvar as figuras? (s/n): ").lower().strip()
            if resp == "s":
                viewer.save_figures(figures_to_save, args.output_dir)
                print("\nFiguras salvas com sucesso!")
        except Exception as e:
            print(f"\nErro ao salvar: {e}")
    
    print("\n" + "=" * 60)
    print("Visualizacao concluida!")
    print("=" * 60)


if __name__ == "__main__":
    main()