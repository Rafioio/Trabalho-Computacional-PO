#!/usr/bin/env python3
"""Pareto frontier experiment: sweep mu with fine granularity on pequena tier."""

import json
import os
import sys
import time
import numpy as np
import gurobipy as gp
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent.parent.parent
SAMPLES_DIR = BASE / "docs" / "pareto" / "samples"
REPORT_PATH = BASE / "docs" / "pareto" / "pareto_report.md"
PLOT_PATH = BASE / "docs" / "pareto" / "pareto_frontier.png"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123]
MU_VALUES = [round(i * 0.1, 1) for i in range(1, 31)]
THETA = 1.0
W = [0.35, 0.15, 0.30, 0.20]
TIER = "pequena"
N, K, Q = 50, 4, 10

WEIGHT_LABEL = "ahp_default"

from src.utils.generate_data import generate_data, parse_args as gen_args
from src.data.loader import save_json
from src.model.function_normalizer import normalize_function
from src.model.solver import build_model
from src.model.objective import build_objective


all_points = []

def run_mu_experiment(seed):
    print(f"\n{'='*60}")
    print(f"Seed {seed} | Gerando dados N={N} K={K} Q={Q}")
    print(f"{'='*60}")

    t0 = time.time()
    args = gen_args([
        "--num-n", str(N), "--num-k", str(K), "--num-q", str(Q),
        "--seed", str(seed), "--output", "/dev/null", "--quiet",
    ])
    rng = np.random.default_rng(seed)
    data = generate_data(args, rng, seed)
    data["W1"], data["W2"], data["W3"], data["W4"] = W[0], W[1], W[2], W[3]
    gen_time = time.time() - t0
    print(f"  Dados gerados em {gen_time:.2f}s")

    t1 = time.time()
    normalize_function(data, verbose=False)
    norm_time = time.time() - t1
    print(f"  Normalizacao em {norm_time:.1f}s")

    print(f"\n  Varrendo mu de {MU_VALUES[0]} a {MU_VALUES[-1]} (theta={THETA})...")
    for mu in MU_VALUES:
        data["mu"] = mu
        data["theta"] = THETA

        t2 = time.time()
        model, vars_dict, domains, obj_exprs = build_model(data)
        obj_exprs = build_objective(model, data, vars_dict, domains)
        model.Params.OutputFlag = 0
        model.Params.MIPGap = 0.01
        model.optimize()
        solve_time = time.time() - t2

        if model.status == gp.GRB.OPTIMAL:
            f1 = obj_exprs["f1"].getValue()
            f2 = obj_exprs["f2"].getValue()
            f3 = obj_exprs["f3"].getValue()
            f4 = obj_exprs["f4"].getValue()
            F_usuario = obj_exprs["F_usuario"].getValue()
            F_operador = obj_exprs["F_operador"].getValue()
            Cad = float(vars_dict["Cad"].X)
            Cap = {int(k): float(vars_dict["Cap"][k].X) for k in data["K"]}
            pontos_ativos = [n for n in data["N"] if vars_dict["x"][n].X > 0.5]
            obj_val = model.objVal

            norm_data = data.get("_normalization", {})
            utopia = norm_data.get("utopia", [0, 0, 0, 0])
            anti = norm_data.get("anti_utopia", [1, 1, 1, 1])
            def nv(v, lo, hi):
                d = hi - lo
                return (v - lo) / d if abs(d) > 1e-12 else 0.0

            summary = {
                "status": "optimal", "objVal": obj_val,
                "mu": mu, "theta": THETA,
                "f1_raw": f1, "f2_raw": f2, "f3_raw": f3, "f4_raw": f4,
                "f1_norm": nv(f1, utopia[0], anti[0]),
                "f2_norm": nv(f2, utopia[1], anti[1]),
                "f3_norm": nv(f3, utopia[2], anti[2]),
                "f4_norm": nv(f4, utopia[3], anti[3]),
                "F_usuario": F_usuario, "F_operador": F_operador,
                "Cad": Cad, "total_buses": sum(Cap.values()) + Cad,
                "active_stops": len(pontos_ativos),
                "route_capacities": Cap,
            }

            solucao = {
                "valor_objetivo": round(obj_val, 4), "status": "optimal",
                "mu": mu, "theta": THETA,
                "num_pontos_ativos": len(pontos_ativos),
                "pontos_ativos": pontos_ativos,
                "capacidade_rotas": Cap,
                "capacidade_adicional": int(Cad),
                "objetivos": {
                    "f1_custo_social": round(f1, 2),
                    "f2_penalidade_espacamento": round(f2, 4),
                    "f3_custo_infraestrutura": round(f3, 2),
                    "f4_viabilidade_tecnica": round(f4, 4),
                    "F_usuario": round(F_usuario, 4),
                    "F_operador": round(F_operador, 4),
                },
            }

            params = {
                "tier": TIER, "seed": seed, "mu": mu, "theta": THETA,
                "NumN": N, "NumK": K, "NumQ": Q,
                "W1": W[0], "W2": W[1], "W3": W[2], "W4": W[3],
                "weight_label": WEIGHT_LABEL,
                "solve_time_s": round(solve_time, 3),
                "timestamp": datetime.now().isoformat(),
            }

            run_label = f"s{seed}_mu{mu}"
            run_dir = SAMPLES_DIR / run_label
            run_dir.mkdir(parents=True, exist_ok=True)

            save_json(data, str(run_dir / "input_data.json"))
            with open(run_dir / "params.json", "w") as f:
                json.dump(params, f, indent=2)
            with open(run_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            with open(run_dir / "solucao.json", "w") as f:
                json.dump(solucao, f, indent=2)

            all_points.append({
                "seed": seed, "mu": mu,
                "F_usuario": F_usuario, "F_operador": F_operador,
                "f1_raw": f1, "f2_raw": f2, "f3_raw": f3, "f4_raw": f4,
                "objVal": obj_val,
                "active_stops": len(pontos_ativos),
            })

            print(f"    mu={mu:.1f}  F_usuario={F_usuario:.4f}  F_operador={F_operador:.4f}  "
                  f"obj={obj_val:.4f}  stops={len(pontos_ativos)}  time={solve_time:.2f}s")
        else:
            print(f"    mu={mu:.1f}  FAILED (status {model.status})")

    print(f"\n  Seed {seed} concluida em {time.time()-t0:.1f}s total")

for seed in SEEDS:
    run_mu_experiment(seed)

print(f"\n{'='*60}")
print(f"Total de pontos coletados: {len(all_points)}")
print(f"{'='*60}")

all_points.sort(key=lambda p: (p["F_usuario"], p["F_operador"]))

pareto = []
for p in all_points:
    dominated = False
    for q in pareto:
        if q["F_usuario"] <= p["F_usuario"] and q["F_operador"] <= p["F_operador"] and \
           (q["F_usuario"] < p["F_usuario"] or q["F_operador"] < p["F_operador"]):
            dominated = True
            break
    if not dominated:
        pareto = [q for q in pareto if not
                  (p["F_usuario"] <= q["F_usuario"] and p["F_operador"] <= q["F_operador"] and
                   (p["F_usuario"] < q["F_usuario"] or p["F_operador"] < q["F_operador"]))]
        pareto.append(p)

pareto.sort(key=lambda p: p["F_usuario"])
print(f"Pontos na fronteira de Pareto: {len(pareto)}")

fig, ax = plt.subplots(figsize=(10, 7))

colors = {42: "#3498db", 123: "#e74c3c"}
for seed in SEEDS:
    pts = [p for p in all_points if p["seed"] == seed]
    xs = [p["F_usuario"] for p in pts]
    ys = [p["F_operador"] for p in pts]
    ax.scatter(xs, ys, c=colors[seed], label=f"Seed {seed}", s=30, alpha=0.6, zorder=2)

if pareto:
    px = [p["F_usuario"] for p in pareto]
    py = [p["F_operador"] for p in pareto]
    ax.scatter(px, py, c="black", s=60, marker="D", zorder=4, label="Fronteira Pareto")
    for p in pareto:
        ax.annotate(f"mu={p['mu']:.1f}", (p["F_usuario"], p["F_operador"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.8)

ax.set_xlabel("F_usuario (custo para o usuário)", fontsize=12)
ax.set_ylabel("F_operador (custo para o operador)", fontsize=12)
ax.set_title(f"Fronteira de Pareto - Tier {TIER} (N={N}, K={K}, Q={Q})\n"
             f"Varrendo mu=[{MU_VALUES[0]},{MU_VALUES[-1]}] theta={THETA} fixo",
             fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, None)
ax.set_ylim(0, None)

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Grafico salvo em {PLOT_PATH}")

now = datetime.now().strftime("%Y-%m-%d %H:%M")

with open(REPORT_PATH, "w") as f:
    f.write("# Relatório da Fronteira de Pareto - SouBuz\n\n")
    f.write(f"**Gerado em:** {now}\n\n")

    f.write("## Configuração do Experimento\n\n")
    f.write(f"| Parâmetro | Valor |\n")
    f.write(f"|-----------|-------|\n")
    f.write(f"| Tier | {TIER} |\n")
    f.write(f"| Nós (N) | {N} |\n")
    f.write(f"| Rotas (K) | {K} |\n")
    f.write(f"| Zonas de demanda (Q) | {Q} |\n")
    f.write(f"| Pesos (W1,W2,W3,W4) | ({W[0]},{W[1]},{W[2]},{W[3]}) {WEIGHT_LABEL} |\n")
    f.write(f"| Sementes | {SEEDS} |\n")
    f.write(f"| Valores de mu | {MU_VALUES[0]} a {MU_VALUES[-1]} (passo 0.1) |\n")
    f.write(f"| theta fixo | {THETA} |\n")
    f.write(f"| Total de runs | {len(all_points)} |\n")
    f.write(f"| Pontos na fronteira Pareto | {len(pareto)} |\n\n")

    f.write("## Resultados\n\n")
    f.write("![Fronteira de Pareto](../pareto_frontier.png)\n\n")

    f.write("## Interpretação\n\n")
    f.write("A fronteira de Pareto mostra o trade-off entre os macro-objetivos:\n\n")
    f.write("- **F_usuario** (eixo X): combina custo social (f1) e penalidade de\n")
    f.write("  espaçamento (f2). Valores menores = melhor para o usuário.\n")
    f.write("- **F_operador** (eixo Y): combina custo de infraestrutura (f3) e\n")
    f.write("  viabilidade técnica (f4). Valores menores = melhor para o operador.\n\n")
    f.write("Cada ponto representa uma solução ótima para um valor específico de mu.\n")
    f.write("Os losangos pretos destacam as soluções não-dominadas (fronteira Pareto).\n\n")
    f.write("### Leitura da Fronteira\n\n")
    f.write("| Região | mu baixo (< 1) | mu ≈ 1 | mu alto (> 1) |\n")
    f.write("|--------|---------------|--------|--------------|\n")
    f.write("| Tendência | Menor F_operador | Equilíbrio | Menor F_usuario |\n")
    f.write("| Quem ganha | Operador | Neutro | Usuário |\n")
    f.write("| Quem perde | Usuário | — | Operador |\n\n")

    f.write("## Pontos da Fronteira de Pareto\n\n")
    f.write("| mu | F_usuario | F_operador | f1 (social) | f2 (spacing) | f3 (infra) | f4 (técnico) | Stops |\n")
    f.write("|----|-----------|------------|-------------|--------------|------------|--------------|-------|\n")
    for p in pareto:
        mu_str = f"{p['mu']:.1f}"
        fu = f"{p['F_usuario']:.4f}"
        fo = f"{p['F_operador']:.4f}"
        f1s = f"{p['f1_raw']:.0f}"
        f2s = f"{p['f2_raw']:.4f}"
        f3s = f"{p['f3_raw']:.0f}"
        f4s = f"{p['f4_raw']:.1f}"
        st = p['active_stops']
        f.write(f"| {mu_str} | {fu} | {fo} | {f1s} | {f2s} | {f3s} | {f4s} | {st} |\n")

    f.write("\n## Todos os Pontos\n\n")
    f.write("| seed | mu | F_usuario | F_operador | objVal | Stops |\n")
    f.write("|------|----|-----------|------------|--------|-------|\n")
    for p in all_points:
        f.write(f"| {p['seed']} | {p['mu']:.1f} | {p['F_usuario']:.4f} | "
                f"{p['F_operador']:.4f} | {p['objVal']:.4f} | {p['active_stops']} |\n")

    f.write("\n## Conclusões\n\n")
    f.write(f"1. A fronteira de Pareto foi construída com {len(pareto)} pontos não-dominados\n")
    f.write(f"   a partir de {len(all_points)} execuções do modelo.\n")
    f.write(f"2. Mu baixo ({MU_VALUES[0]}-1.0) favorece o operador (menor F_operador).\n")
    f.write(f"3. Mu alto (1.0-{MU_VALUES[-1]}) favorece o usuário (menor F_usuario).\n")
    f.write(f"4. O joelho da curva indica a região de melhor equilíbrio.\n")
    f.write(f"5. A normalização Min-Max permite comparar diretamente os valores.\n\n")

print(f"  Relatorio salvo em {REPORT_PATH}")
print("Done!")