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
"""

import json
import math
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D


def ler_dados_json(path="dados_generated.json"):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Erro: arquivo '{path}' nao encontrado.")
        return None


def ler_solucao_json(path="solucao.json"):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _limits(posicoes, padding=0.12):
    xs = [p[0] for p in posicoes]
    ys = [p[1] for p in posicoes]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    dx = (xmax - xmin) or 1
    dy = (ymax - ymin) or 1
    return xmin - dx * padding, xmax + dx * padding, ymin - dy * padding, ymax + dy * padding


def _grid(ax, xmin, xmax, ymin, ymax):
    nx = max(5, min(10, int((xmax - xmin) / 300)))
    ny = max(5, min(10, int((ymax - ymin) / 300)))
    for x in [xmin + i * (xmax - xmin) / nx for i in range(nx + 1)]:
        ax.axvline(x, color="lightgray", ls="--", lw=0.5, alpha=0.5, zorder=0)
    for y in [ymin + i * (ymax - ymin) / ny for i in range(ny + 1)]:
        ax.axhline(y, color="lightgray", ls="--", lw=0.5, alpha=0.5, zorder=0)


def _scale_bar(ax, xmin, xmax, ymin, ymax):
    w = xmax - xmin
    h = ymax - ymin
    length = w * 0.15
    for unit in [500, 200, 100, 50, 25, 10, 5]:
        if length >= unit * 1.5:
            length = round(length / unit) * unit
            break
    x0 = xmax - w * 0.12
    x1 = x0 + length
    y0 = ymin + h * 0.05
    ax.plot([x0, x1], [y0, y0], "k-", lw=3, zorder=10)
    ax.plot([x0, x0], [y0 - h * 0.008, y0 + h * 0.008], "k-", lw=2, zorder=10)
    ax.plot([x1, x1], [y0 - h * 0.008, y0 + h * 0.008], "k-", lw=2, zorder=10)
    ax.text(
        (x0 + x1) / 2, y0 - h * 0.015, f"{int(length)} m",
        ha="center", va="top", fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, edgecolor="gray"),
    )


def _compass(ax, xmin, xmax, ymin, ymax):
    w = xmax - xmin
    h = ymax - ymin
    cx = xmax - w * 0.08
    cy = ymax - h * 0.08
    r = min(w, h) * 0.04
    ax.add_patch(Circle((cx, cy), r, fill=False, ec="#333", lw=1.5, zorder=10))
    ax.plot([cx, cx], [cy - r, cy + r], "#333", lw=1.2, zorder=10)
    ax.plot([cx - r, cx + r], [cy, cy], "#333", lw=1.2, zorder=10)
    ax.annotate("N", (cx, cy + r), textcoords="offset points", xytext=(0, 3),
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333")
    for label, dx, dy in [("S", 0, -3), ("L", 3, 0), ("O", -3, 0)]:
        ax.annotate(label, (cx + dx * r / r * 0 if label in ("S",) else cx + (r if label == "L" else -r), cy),
                    textcoords="offset points", xytext=(dx, dy) if label != "O" else (-3, 0),
                    ha="center" if label in ("N", "S") else ("left" if label == "L" else "right"),
                    va="center", fontsize=8, color="#333")


def visualizar_pontos(dados, solucao=None):
    if dados is None:
        return None, None

    vis = dados.get("visualizacao", {})
    stats = dados.get("estatisticas", {})
    d_walk_max = dados.get("d_walk_max", 500)

    pos_pontos = vis.get("posicoes_pontos", [])
    pos_dem = vis.get("posicoes_demandas", [])
    qualidades = vis.get("qualidades_pontos", [])
    demandas = vis.get("niveis_demanda", [])

    tem_sol = solucao is not None
    pontos_ativos = set(solucao.get("pontos_ativos", [])) if tem_sol else set()

    all_pos = pos_pontos + pos_dem
    xmin, xmax, ymin, ymax = _limits(all_pos, 0.12)

    ar = (xmax - xmin) / ((ymax - ymin) or 1)
    fw, fh = (14, 14 / ar) if ar > 0 else (14, 10)
    fig, ax = plt.subplots(1, 1, figsize=(fw, fh))
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xlabel="Coordenada X (m)", ylabel="Coordenada Y (m)")
    ax.set_facecolor("#f8f9fa")
    _grid(ax, xmin, xmax, ymin, ymax)

    rng_x = xmax - xmin
    sz_base = max(80, min(300, rng_x * 0.03))
    sz_dem = max(30, min(150, rng_x * 0.02))
    max_dem = max(demandas) if demandas else 1

    for i, (x, y) in enumerate(pos_dem):
        s = sz_dem + (demandas[i] / max_dem) * sz_dem
        ax.scatter(x, y, s=s, c="#87CEEB", alpha=0.7, ec="#1E90FF", lw=1.5, zorder=2)
        if len(pos_dem) <= 30:
            ax.annotate(f"D{i+1}", (x, y), textcoords="offset points", xytext=(5, 5),
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7), zorder=3)

    for i, (x, y) in enumerate(pos_pontos):
        ativo = (i + 1) in pontos_ativos
        q = qualidades[i] if i < len(qualidades) else 0.5
        s = sz_base + q * sz_base * 0.5
        if ativo and tem_sol:
            c, ec, mk = "#2E8B57", "#1a5a38", "s"
        else:
            c, ec, mk = (max(0.2, 1 - q), max(0.3, q), 0.2), "#333", "o"
        ax.scatter(x, y, s=s, c=[c], marker=mk, ec=ec, lw=2, zorder=4, alpha=0.95 if ativo else 0.85)
        ax.annotate(str(i + 1), (x, y), ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white", zorder=5)

    if pos_dem:
        cx, cy = pos_dem[0]
        circ = Circle((cx, cy), d_walk_max, fill=False, ec="#1E90FF", ls="--", lw=1.5, alpha=0.4, zorder=1)
        ax.add_patch(circ)
        ax.text(cx + d_walk_max * 0.6, cy + d_walk_max * 0.6,
                f"Raio de acessibilidade\n{d_walk_max:.0f} m",
                fontsize=8, alpha=0.6, ha="center",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    _scale_bar(ax, xmin, xmax, ymin, ymax)
    _compass(ax, xmin, xmax, ymin, ymax)

    n_pontos = len(pos_pontos)
    n_ativos = len(pontos_ativos)
    n_dem = len(pos_dem)
    if tem_sol:
        titulo = f"SOLUCAO DO MODELO DE OTIMIZACAO\n{n_pontos} Pontos | {n_ativos} Ativos | {n_dem} Demandas"
        titulo += f"\nValor Objetivo: {solucao.get('valor_objetivo', 0):.2f}"
    else:
        titulo = f"DADOS DE ENTRADA (PRE-OTIMIZACAO)\n{n_pontos} Pontos Candidatos | {n_dem} Nos de Demanda"
    ax.set_title(titulo, fontsize=14, fontweight="bold", pad=20)

    info = (
        f"PARAMETROS DO MODELO:\n"
        f"\u2022 d_walk_max = {d_walk_max:.0f} m\n"
        f"\u2022 d_route_max = {dados.get('d_route_max', 800):.0f} m\n"
        f"\u2022 Capt = {dados.get('Capt', 800)}\n"
        f"\u2022 m_max = {dados.get('m_max', 3)} rotas/ponto\n"
        f"\u2022 \u03c9 = {dados.get('omega', 50)}\n"
        f"\u2022 Demanda total = {stats.get('demanda_total', 0)}"
    )
    ax.text(xmin, ymax, info, transform=ax.transData, fontsize=9,
            verticalalignment="top", horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="#FFFFE0", alpha=0.9, edgecolor="#CCC"), zorder=10)

    if tem_sol:
        leg = [
            Line2D([0], [0], marker="o", color="w", mfc="#87CEEB", ms=10,
                   label="Nos de Demanda", mec="#1E90FF", mew=1.5),
            Line2D([0], [0], marker="o", color="w", mfc="gray", ms=10,
                   label="Ponto (nao ativo)", mec="#333"),
            Line2D([0], [0], marker="s", color="w", mfc="#2E8B57", ms=10,
                   label="Ponto Ativo (selecionado)", mec="#1a5a38"),
            Line2D([0], [0], ls="--", color="#1E90FF", lw=1.5, alpha=0.6,
                   label=f"Raio de Acessibilidade ({d_walk_max:.0f}m)"),
        ]
    else:
        leg = [
            Line2D([0], [0], marker="o", color="w", mfc="#87CEEB", ms=10,
                   label="Nos de Demanda", mec="#1E90FF", mew=1.5),
            Line2D([0], [0], marker="o", color="w", mfc=(0.6, 0.6, 0.2), ms=10,
                   label="Ponto Candidato (cor = qualidade)", mec="#333"),
            Line2D([0], [0], ls="--", color="#1E90FF", lw=1.5, alpha=0.6,
                   label=f"Raio de Acessibilidade ({d_walk_max:.0f}m)"),
        ]
    ax.legend(handles=leg, loc="lower left", fontsize=9, framealpha=0.9, edgecolor="#CCC")
    fig.tight_layout()
    return fig, ax


def visualizar_densidade(dados):
    if dados is None:
        return None, None
    vis = dados.get("visualizacao", {})
    stats = dados.get("estatisticas", {})
    pos_pontos = vis.get("posicoes_pontos", [])
    pos_dem = vis.get("posicoes_demandas", [])
    qualidades = vis.get("qualidades_pontos", [])
    demandas = vis.get("niveis_demanda", [])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    xs = [p[0] for p in pos_pontos]
    ys = [p[1] for p in pos_pontos]
    if xs:
        hb = ax.hexbin(xs, ys, gridsize=25, cmap="YlOrRd", alpha=0.7, mincnt=1)
        ax.scatter(xs, ys, c="blue", s=20, alpha=0.5)
        plt.colorbar(hb, ax=ax, label="Densidade")
    ax.set_title("Densidade dos Pontos Candidatos", fontsize=12, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    ax = axes[0, 1]
    xs = [p[0] for p in pos_dem]
    ys = [p[1] for p in pos_dem]
    if xs:
        hb = ax.hexbin(xs, ys, gridsize=25, cmap="Blues", alpha=0.7, mincnt=1)
        ax.scatter(xs, ys, c="red", s=20, alpha=0.5)
        plt.colorbar(hb, ax=ax, label="Densidade")
    ax.set_title("Densidade dos Nos de Demanda", fontsize=12, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    ax = axes[1, 0]
    if qualidades:
        ax.hist(qualidades, bins=20, color="green", alpha=0.7, edgecolor="black")
        ax.axvline(stats.get("qualidade_media", 0), color="red", ls="--",
                   label=f"Media: {stats.get('qualidade_media', 0):.3f}")
        ax.legend()
    ax.set_title("Distribuicao da Qualidade dos Pontos (w[n])", fontsize=12, fontweight="bold")
    ax.set_xlabel("Qualidade")
    ax.set_ylabel("Frequencia")

    ax = axes[1, 1]
    if demandas:
        ax.hist(demandas, bins=20, color="blue", alpha=0.7, edgecolor="black")
        ax.axvline(stats.get("demanda_media", 0), color="red", ls="--",
                   label=f"Media: {stats.get('demanda_media', 0):.1f}")
        ax.legend()
    ax.set_title("Distribuicao dos Niveis de Demanda (de[q])", fontsize=12, fontweight="bold")
    ax.set_xlabel("Demanda (passageiros)")
    ax.set_ylabel("Frequencia")

    seed = dados.get("metadata", {}).get("semente", "?")
    fig.suptitle(f"Analise Estatistica dos Dados (semente: {seed})",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig, axes


def visualizar_comparativo(dados, solucao):
    if dados is None or solucao is None:
        return None, None
    vis = dados.get("visualizacao", {})
    pos_pontos = vis.get("posicoes_pontos", [])
    qualidades = vis.get("qualidades_pontos", [])
    pontos_ativos = set(solucao.get("pontos_ativos", []))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    for i, (x, y) in enumerate(pos_pontos):
        q = qualidades[i] if i < len(qualidades) else 0.5
        c = (max(0.2, 1 - q), max(0.3, q), 0.2)
        ax1.scatter(x, y, s=200, c=[c], marker="o", ec="#333", lw=2, alpha=0.8)
        ax1.annotate(str(i + 1), (x, y), ha="center", va="center", fontsize=8, fontweight="bold", color="white")
    ax1.set_title(f"ANTES: {len(pos_pontos)} Pontos Candidatos", fontsize=12, fontweight="bold")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")

    for i, (x, y) in enumerate(pos_pontos):
        ativo = (i + 1) in pontos_ativos
        if ativo:
            ax2.scatter(x, y, s=250, c=["#2E8B57"], marker="s", ec="#1a5a38", lw=2.5, alpha=0.95)
            ax2.annotate(str(i + 1), (x, y), ha="center", va="center", fontsize=9,
                         fontweight="bold", color="white")
        else:
            ax2.scatter(x, y, s=100, c=["lightgray"], marker="o", ec="gray", lw=1, alpha=0.5)
            ax2.annotate(str(i + 1), (x, y), ha="center", va="center", fontsize=7, color="gray")
    ax2.set_title(f"DEPOIS: {len(pontos_ativos)} Pontos Ativos Selecionados", fontsize=12, fontweight="bold")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")

    xmin, xmax, ymin, ymax = _limits(pos_pontos, 0.1)
    ax1.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    ax2.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

    fig.suptitle(f"Comparacao Antes/Depois da Otimizacao\nValor Objetivo: {solucao.get('valor_objetivo', 0):.2f}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig, (ax1, ax2)


def main():
    print("=" * 60)
    print("VISUALIZADOR DO SISTEMA DE PARADAS DE ONIBUS")
    print("=" * 60)

    path = sys.argv[1] if len(sys.argv) > 1 else "dados_generated.json"
    print(f"\nCarregando '{path}'...")
    dados = ler_dados_json(path)
    if dados is None:
        return

    solucao = ler_solucao_json("solucao.json")

    print(f"\nDados carregados:")
    print(f"  - Pontos candidatos: {dados.get('NumN', '?')}")
    print(f"  - Rotas: {dados.get('NumK', '?')}")
    print(f"  - Nos de demanda: {dados.get('NumQ', '?')}")
    stats = dados.get("estatisticas", {})
    print(f"  - Demanda total: {stats.get('demanda_total', '?')} passageiros")
    print(f"  - Qualidade media dos pontos: {stats.get('qualidade_media', '?'):.3f}")
    print(f"  - d_walk_max: {dados.get('d_walk_max', 500)} m")

    if solucao:
        print(f"\nSolucao encontrada:")
        print(f"  - Valor objetivo: {solucao.get('valor_objetivo', 0):.2f}")
        print(f"  - Pontos ativos: {len(solucao.get('pontos_ativos', []))}")

    print("\nGerando visualizacao principal...")
    fig1, _ = visualizar_pontos(dados, solucao)

    print("Gerando analise de densidade...")
    fig2, _ = visualizar_densidade(dados)

    if solucao:
        print("Gerando visualizacao comparativa...")
        fig3, _ = visualizar_comparativo(dados, solucao)

    plt.show()

    try:
        resp = input("\nDeseja salvar as figuras? (s/n): ").lower().strip()
        if resp == "s":
            fig1.savefig("visualizacao_pontos.png", dpi=150, bbox_inches="tight", facecolor="white")
            fig2.savefig("visualizacao_densidade.png", dpi=150, bbox_inches="tight", facecolor="white")
            if solucao:
                fig3.savefig("visualizacao_comparativo.png", dpi=150, bbox_inches="tight", facecolor="white")
            print("\nFiguras salvas:")
            print("  - visualizacao_pontos.png")
            print("  - visualizacao_densidade.png")
            if solucao:
                print("  - visualizacao_comparativo.png")
    except Exception as e:
        print(f"\nErro ao salvar: {e}")

    print("\n" + "=" * 60)
    print("Visualizacao concluida!")
    print("=" * 60)


if __name__ == "__main__":
    main()