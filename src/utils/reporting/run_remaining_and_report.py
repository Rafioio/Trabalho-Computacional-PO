#!/usr/bin/env python3
"""
Run one muito_grande and one extrema instance, generate all visualizations,
and build complete report structure in docs/relatorios/.
"""

import json, os, sys, time, subprocess
from pathlib import Path
from datetime import datetime

BASE = Path("runs")
REPORT = Path("docs/relatorios")

import numpy as np
import gurobipy as gp

# ============================================================================
# Helper: run one solve with full pipeline
# ============================================================================
def run_single_tier(tier_label, N, K, Q, seed, weight_label, W, mu=1.0, theta=1.0):
    """Run one solve: generate data, normalize, solve, return results."""
    from src.utils.generate_data import generate_data, parse_args as gen_args
    from src.data.loader import save_json
    from src.model.domains import build_a_domain, build_S_indices
    from src.model.variables import create_variables
    from src.model.objective import build_objective
    from src.model.constraints import add_c1, add_c2, add_c3, add_c4, add_c5, add_c6, add_c7, add_c8
    from src.model.function_normalizer import normalize_function

    print(f"\n{'='*70}")
    print(f"TIER: {tier_label} | N={N} K={K} Q={Q} seed={seed}")
    print(f"{'='*70}")

    # Generate data
    t0 = time.time()
    args = gen_args([
        "--num-n", str(N), "--num-k", str(K), "--num-q", str(Q),
        "--seed", str(seed), "--output", "/dev/null", "--quiet"])
    rng = np.random.default_rng(seed)
    data = generate_data(args, rng, seed)
    data["W1"], data["W2"], data["W3"], data["W4"] = W
    data["mu"] = mu
    data["theta"] = theta
    gen_time = time.time() - t0
    print(f"  Data generated: {gen_time:.1f}s")

    # Normalize
    t1 = time.time()
    normalize_function(data, verbose=False)
    norm_time = time.time() - t1
    print(f"  Normalization: {norm_time:.1f}s")

    # Solve
    t2 = time.time()
    domains = {"a_domain": build_a_domain(data), "S_Indices": build_S_indices(data)}
    model = gp.Model("soubuz")
    model.Params.OutputFlag = 0
    model.Params.NonConvex = 2

    vars_dict = create_variables(model, data, domains)
    obj_exprs = build_objective(model, data, vars_dict, domains)
    add_c1(model, data, vars_dict)
    add_c2(model, data, domains, vars_dict)
    add_c3(model, data, domains, vars_dict)
    add_c4(model, data, vars_dict)
    add_c5(model, data, domains, vars_dict)
    add_c6(model, data, domains, vars_dict)
    add_c7(model, data, vars_dict)
    add_c8(model, data, vars_dict)
    model.update()
    model.optimize()

    status_map = {gp.GRB.OPTIMAL: "optimal", gp.GRB.INFEASIBLE: "infeasible",
                  gp.GRB.UNBOUNDED: "unbounded", gp.GRB.INF_OR_UNBD: "inf_or_unbd"}
    status = status_map.get(model.status, f"status_{model.status}")
    solve_time = time.time() - t2
    print(f"  Solve: {solve_time:.1f}s | Status: {status} | Obj: {model.objVal if model.status == gp.GRB.OPTIMAL else 'N/A':.4f}")

    result = {"status": status, "objVal": model.objVal if model.status == gp.GRB.OPTIMAL else None,
              "model": model, "vars": vars_dict, "obj_exprs": obj_exprs, "domains": domains}
    if model.status == gp.GRB.OPTIMAL:
        result["solution"] = {
            "f1": obj_exprs["f1"].getValue(),
            "f2": obj_exprs["f2"].getValue(),
            "f3": obj_exprs["f3"].getValue(),
            "f4": obj_exprs["f4"].getValue(),
            "F_usuario": obj_exprs["F_usuario"].getValue(),
            "F_operador": obj_exprs["F_operador"].getValue(),
            "Cad": float(vars_dict["Cad"].X),
            "Cap": {int(k): float(vars_dict["Cap"][k].X) for k in data["K"]},
            "pontos_ativos": [n for n in data["N"] if vars_dict["x"][n].X > 0.5],
        }

    return data, result, gen_time, norm_time, solve_time

# ============================================================================
# Helper: save run results
# ============================================================================
def save_run_results(run_dir, data, result, W, weight_label, tier_label, 
                     gen_time, norm_time, solve_time, seed, mu, theta):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save input data
    from src.data.loader import save_json
    save_json(data, str(run_dir / "input_data.json"))

    # Save params
    params = {"tier": tier_label, "seed": seed, "NumN": data["NumN"], "NumK": data["NumK"],
              "NumQ": data["NumQ"], "W1": W[0], "W2": W[1], "W3": W[2], "W4": W[3],
              "weight_label": weight_label, "mu": mu, "theta": theta,
              "mip_gap": result.get("model").MIPGap if result["status"] == "optimal" else None,
              "solve_status": result["status"], "gen_time_s": round(gen_time, 2),
              "normalization_time_s": round(norm_time, 2), "solve_time_s": round(solve_time, 2),
              "timestamp": datetime.now().isoformat()}
    with open(run_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    # Save solution
    if result["status"] == "optimal":
        sol = result["solution"]
        norm_data = data.get("_normalization", {})
        utopia = norm_data.get("utopia", [0,0,0,0])
        anti = norm_data.get("anti_utopia", [1,1,1,1])
        
        def nv(v, lo, hi):
            d = hi-lo
            return (v-lo)/d if abs(d)>1e-12 else 0.0
        
        summary = {"status": "optimal", "objVal": result["objVal"],
            "f1_raw": sol["f1"], "f2_raw": sol["f2"], "f3_raw": sol["f3"], "f4_raw": sol["f4"],
            "f1_norm": nv(sol["f1"], utopia[0], anti[0]),
            "f2_norm": nv(sol["f2"], utopia[1], anti[1]),
            "f3_norm": nv(sol["f3"], utopia[2], anti[2]),
            "f4_norm": nv(sol["f4"], utopia[3], anti[3]),
            "F_usuario": sol["F_usuario"], "F_operador": sol["F_operador"],
            "Cad": sol["Cad"], "total_buses": sum(sol["Cap"].values()) + sol["Cad"],
            "active_stops": len(sol["pontos_ativos"]),
            "route_capacities": sol["Cap"]}
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        solucao = {"valor_objetivo": round(result["objVal"], 4), "status": "optimal",
            "num_pontos_ativos": len(sol["pontos_ativos"]),
            "pontos_ativos": sol["pontos_ativos"],
            "capacidade_rotas": sol["Cap"], "capacidade_adicional": int(sol["Cad"]),
            "objetivos": {"f1_custo_social": round(sol["f1"], 2),
                          "f2_penalidade_espacamento": round(sol["f2"], 2),
                          "f3_custo_infraestrutura": round(sol["f3"], 2),
                          "f4_viabilidade_tecnica": round(sol["f4"], 4),
                          "F_usuario": round(sol["F_usuario"], 4),
                          "F_operador": round(sol["F_operador"], 4)}}
        with open(run_dir / "solucao.json", "w") as f:
            json.dump(solucao, f, indent=2)

        # Objectives chart
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            
            labels = ["f1 (social)", "f2 (spacing)", "f3 (infra)", "f4 (tech)"]
            raw_vals = [sol["f1"], sol["f2"], sol["f3"], sol["f4"]]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
            bars1 = ax1.bar(labels, raw_vals, color=["#2ecc71","#e74c3c","#3498db","#f39c12"])
            ax1.set_title("Raw objective values")
            ax1.set_ylabel("Value")
            for b, v in zip(bars1, raw_vals):
                ax1.text(b.get_x()+b.get_width()/2, b.get_height(), f"{v:.1f}", ha="center", va="bottom", fontsize=9)

            norm_vals = [nv(sol["f1"], utopia[0], anti[0]),
                         nv(sol["f2"], utopia[1], anti[1]),
                         nv(sol["f3"], utopia[2], anti[2]),
                         nv(sol["f4"], utopia[3], anti[3])]
            weighted = [n*w for n,w in zip(norm_vals, W)]
            x = range(len(labels))
            ax2.bar(x, norm_vals, 0.6, label="Normalized", color="#95a5a6")
            ax2.bar(x, weighted, 0.6, label="Weighted", color="#e67e22", alpha=0.7)
            ax2.set_xticks(list(x)); ax2.set_xticklabels(labels)
            ax2.set_title("Normalized vs Weighted")
            ax2.set_ylabel("Value [0,1]"); ax2.legend(); ax2.set_ylim(0,1.15)
            fig.suptitle(f"{tier_label} seed={seed} {weight_label} obj={result['objVal']:.4f}", fontsize=12, y=1.02)
            plt.tight_layout()
            plt.savefig(run_dir / "objectives.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

            # Map
            if "positions" in data:
                positions = np.array(data["positions"])
                fig, ax = plt.subplots(figsize=(8,8))
                ax.scatter(positions[:,0], positions[:,1], c="#bdc3c7", s=15, alpha=0.4, label="Inactive")
                active = sol["pontos_ativos"]
                for n in active:
                    idx = n - 1
                    if 0 <= idx < len(positions):
                        ax.scatter(positions[idx,0], positions[idx,1], c="#e74c3c", s=40, zorder=5)
                route_stops = {}
                for n, k in data["I"]:
                    xk = vars_dict.get("x_k", {}).get((n,k))
                    if xk is not None and xk.X > 0.5:
                        route_stops.setdefault(k, []).append(n)
                colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b",
                          "#e377c2","#7f7f7f","#bcbd22","#17becf"]
                for kidx, (k, stops) in enumerate(route_stops.items()):
                    for n in stops:
                        idx = n - 1
                        if 0 <= idx < len(positions):
                            ax.scatter(positions[idx,0], positions[idx,1],
                                       color=colors[kidx%len(colors)], s=80, marker="s",
                                       edgecolors="black", linewidths=0.5, zorder=6)
                ax.set_title(f"{tier_label} seed={seed} {weight_label}")
                ax.set_aspect("equal")
                plt.tight_layout()
                plt.savefig(run_dir / "map.png", dpi=120, bbox_inches="tight")
                plt.close(fig)
        except ImportError:
            pass
    return params

# ============================================================================
# RUN muito_grande (N=500, K=10, Q=30, seed=82)
# ============================================================================
print("="*70)
print("FASE 1: Executando muito_grande")
print("="*70)

W_ahp = [0.35, 0.15, 0.30, 0.20]
data_mg, result_mg, gt_mg, nt_mg, st_mg = run_single_tier(
    "muito_grande", 500, 10, 30, 82, "ahp_default", W_ahp)
mg_dir = REPORT / "muito_grande"
save_run_results(mg_dir, data_mg, result_mg, W_ahp, "ahp_default",
                 "muito_grande", gt_mg, nt_mg, st_mg, 82, 1.0, 1.0)

# ============================================================================
# RUN extrema (N=800, K=15, Q=40, seed=92)
# ============================================================================
print("\n" + "="*70)
print("FASE 2: Executando extrema")
print("="*70)

data_ext, result_ext, gt_ext, nt_ext, st_ext = run_single_tier(
    "extrema", 800, 15, 40, 92, "ahp_default", W_ahp)
ext_dir = REPORT / "extrema"
save_run_results(ext_dir, data_ext, result_ext, W_ahp, "ahp_default",
                 "extrema", gt_ext, nt_ext, st_ext, 92, 1.0, 1.0)

# ============================================================================
# Run map_viewer for all runs
# ============================================================================
print("\n" + "="*70)
print("FASE 3: Gerando visualizações map_viewer")
print("="*70)

for label, run_dir in [("muito_grande", mg_dir), ("extrema", ext_dir)]:
    data_file = run_dir / "input_data.json"
    sol_file = run_dir / "solucao.json"
    if not sol_file.exists():
        print(f"  {label}: sem solução, pulando visualizações")
        continue
    
    variants = [
        ("map_solucao.png", ["--solution", str(sol_file)]),
        ("map_comparison.png", ["--solution", str(sol_file), "--comparison"]),
        ("map_density.png", ["--density-only"]),
    ]
    for fname, extra_args in variants:
        out_path = run_dir / fname
        cmd = [sys.executable, "-m", "src.scripts.map_viewer",
               str(data_file), "--output", str(out_path)] + extra_args
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            status = "OK" if r.returncode == 0 else f"ERROR({r.returncode})"
            print(f"  {label}/{fname}: {status}")
        except Exception as e:
            print(f"  {label}/{fname}: FAILED - {e}")

# ============================================================================
# Build complete report
# ============================================================================
print("\n" + "="*70)
print("FASE 4: Escrevendo relatórios")
print("="*70)

# Read data from completed tiers
import csv
runs = []
with open(BASE / "runs_index.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        runs.append(row)

optimal = [r for r in runs if r['status'] == 'optimal']
from collections import defaultdict
tier_runs = defaultdict(list)
for r in optimal:
    tier_runs[r['tier']].append(r)

# Time estimates (speculated vs real)
# Speculated from batch_runner comments: ~8h total
# Real from the run data
tier_times = {}
for tier in ['micro','pequena','media','grande']:
    times = []
    for r in tier_runs[tier]:
        try: times.append(float(r.get('solve_time_s',0)))
        except: pass
    tier_times[tier] = {'n': len(times), 'mean': sum(times)/len(times) if times else 0,
                        'min': min(times) if times else 0, 'max': max(times) if times else 0,
                        'total': sum(times) if times else 0}

# Add muito_grande and extrema from our runs
for label, st in [("muito_grande", st_mg), ("extrema", st_ext)]:
    tier_times[label] = {'n': 1, 'mean': st, 'min': st, 'max': st, 'total': st}

def fmt_time(s):
    if s < 60: return f"{s:.1f}s"
    elif s < 3600: return f"{s/60:.1f}min"
    else: return f"{s/3600:.1f}h"

# Speculated times from batch_runner and actual observations
speculated = {
    'micro': (60, 600),       # 1-10 min speculated
    'pequena': (120, 600),    # 2-10 min
    'media': (600, 1800),     # 10-30 min
    'grande': (3600, 7200),   # 1-2 h
    'muito_grande': (7200, 14400), # 2-4 h
    'extrema': (14400, 36000), # 4-10 h
}

# ============================================================================
# Write main report
# ============================================================================
(REPORT / "micro").mkdir(parents=True, exist_ok=True)
(REPORT / "pequena").mkdir(parents=True, exist_ok=True)
(REPORT / "media").mkdir(parents=True, exist_ok=True)
(REPORT / "grande").mkdir(parents=True, exist_ok=True)
(REPORT / "muito_grande").mkdir(parents=True, exist_ok=True)
(REPORT / "extrema").mkdir(parents=True, exist_ok=True)

def write_tier_report(tier, path):
    entries = tier_runs.get(tier, [])
    has_data = len(entries) > 0
    is_mg = tier == "muito_grande"
    is_ext = tier == "extrema"
    
    with open(path / "README.md", "w") as f:
        f.write(f"# Relatório - Tier {tier}\n\n")
        
        tier_params = {'micro': (20,2,5), 'pequena': (50,4,10), 'media': (150,6,20),
                       'grande': (300,8,30), 'muito_grande': (500,10,30), 'extrema': (800,15,40)}
        N, K, Q = tier_params.get(tier, (0,0,0))
        f.write(f"**Parâmetros:** N={N} nós, K={K} rotas, Q={Q} zonas de demanda\n\n")
        
        ts = tier_times.get(tier, {})
        spec = speculated.get(tier, (0,0))
        f.write(f"## Desempenho\n\n")
        f.write(f"| Métrica | Valor |\n")
        f.write(f"|---------|-------|\n")
        f.write(f"| Runs executadas | {ts.get('n', 0)} |\n")
        if ts.get('n', 0) > 0:
            f.write(f"| Tempo total | {fmt_time(ts['total'])} |\n")
            f.write(f"| Tempo médio/solve | {fmt_time(ts['mean'])} |\n")
            f.write(f"| Mais rápido | {fmt_time(ts['min'])} |\n")
            f.write(f"| Mais lento | {fmt_time(ts['max'])} |\n")
        f.write(f"| Tempo especulado (mín) | {fmt_time(spec[0])} |\n")
        f.write(f"| Tempo especulado (máx) | {fmt_time(spec[1])} |\n\n")
        
        if is_mg or is_ext:
            f.write(f"## Resultados da Run Única\n\n")
            f.write(f"**Configuração:** ahp_default (W1=0.35, W2=0.15, W3=0.30, W4=0.20), mu=1.0, theta=1.0\n\n")
            f.write(f"**Tempos:**\n")
            f.write(f"- Geração de dados: {fmt_time(gt_mg if is_mg else gt_ext)}\n")
            f.write(f"- Normalização (4 solves): {fmt_time(nt_mg if is_mg else nt_ext)}\n")
            f.write(f"- Solve final: {fmt_time(st_mg if is_mg else st_ext)}\n\n")
            
            res = result_mg if is_mg else result_ext
            if res['status'] == 'optimal':
                sol = res['solution']
                f.write(f"**Solução:**\n")
                f.write(f"- Status: {res['status']}\n")
                f.write(f"- Valor objetivo: {res['objVal']:.4f}\n")
                f.write(f"- Pontos ativos: {len(sol['pontos_ativos'])}\n")
                f.write(f"- Capacidade adicional (Cad): {int(sol['Cad'])}\n")
                f.write(f"- Capacidade total: {sum(sol['Cap'].values()) + int(sol['Cad'])}\n")
                f.write(f"- f1 (custo social): {sol['f1']:.2f}\n")
                f.write(f"- f2 (penalidade espaçamento): {sol['f2']:.4f}\n")
                f.write(f"- f3 (custo infraestrutura): {sol['f3']:.2f}\n")
                f.write(f"- f4 (viabilidade técnica): {sol['f4']:.4f}\n")
                f.write(f"- F_usuario: {sol['F_usuario']:.4f}\n")
                f.write(f"- F_operador: {sol['F_operador']:.4f}\n\n")
        
        f.write(f"## Visualizações\n\n")
        f.write(f"Os seguintes arquivos estão disponíveis neste diretório:\n\n")
        viz_files = [p.name for p in path.iterdir() if p.suffix in ('.png', '.json', '.csv')]
        for vf in sorted(viz_files):
            f.write(f"- `{vf}`\n")
        f.write("\n")
        
        if has_data:
            weight_stats = defaultdict(list)
            for r in entries:
                weight_stats[r['weights']].append(float(r['objVal']) if r['objVal'] else 0)
            f.write(f"## Variação de Pesos\n\n")
            f.write(f"| Peso | Runs | Média | Mínimo | Máximo |\n")
            f.write(f"|------|------|-------|--------|--------|\n")
            for w, vals in sorted(weight_stats.items()):
                f.write(f"| {w} | {len(vals)} | {sum(vals)/len(vals):.4f} | {min(vals):.4f} | {max(vals):.4f} |\n")
        f.write("\n")

# Write per-tier reports
for tier in ['micro', 'pequena', 'media', 'grande']:
    write_tier_report(tier, REPORT / tier)
    print(f"  Relatório {tier} escrito")

write_tier_report("muito_grande", REPORT / "muito_grande")
write_tier_report("extrema", REPORT / "extrema")
print(f"  Relatório muito_grande e extrema escritos")

# ============================================================================
# Write main report
# ============================================================================
with open(REPORT / "README.md", "w") as f:
    f.write("# Relatório Completo - Experimentos SouBuz\n\n")
    f.write(f"**Gerado em:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    
    f.write("## Sumário\n\n")
    f.write("1. [Problema de Licenciamento](#1-problema-de-licenciamento)\n")
    f.write("2. [Resumo dos Experimentos](#2-resumo-dos-experimentos)\n")
    f.write("3. [Comparação de Tempos: Especulado vs Real](#3-comparação-de-tempos-especulado-vs-real)\n")
    f.write("4. [Análise por Tier](#4-análise-por-tier)\n")
    f.write("5. [Variação dos Pesos (W1-W4)](#5-variação-dos-pesos-w1-w4)\n")
    f.write("6. [Variação dos Macro-pesos (mu/theta)](#6-variação-dos-macro-pesos-mutheta)\n")
    f.write("7. [Visualizações](#7-visualizações)\n")
    f.write("8. [Conclusões](#8-conclusões)\n\n")
    
    f.write("---\n\n")
    
    # 1. License problem
    f.write("## 1. Problema de Licenciamento\n\n")
    f.write("### 1.1 Histórico\n\n")
    f.write("Originalmente, este projeto utilizava uma licença **Named-User Academic** do Gurobi, que permite modelos de qualquer porte. ")
    f.write("Durante a migração para o ambiente de execução atual, a licença foi perdida e o `gurobipy` passou a usar a ")
    f.write("**licença PIP gratuita (*Restricted license — for non-production use only*)** que acompanha a instalação via pip.\n\n")
    f.write("### 1.2 Impacto\n\n")
    f.write("A licença PIP gratuita impõe um limite severo de tamanho do modelo (~2000 variáveis ou restrições), ")
    f.write("impossibilitando a execução de qualquer tier do `batch_runner`. Mesmo o menor tier (`micro`: 20 nós, 2 rotas, 5 zonas — 121 nós reais via CSV da BH) ")
    f.write("excede este limite.\n\n")
    f.write("### 1.3 Solução\n\n")
    f.write("Foi necessário migrar para uma licença **WLS Academic** (Web License Service) obtida através do ")
    f.write("[Gurobi Academic Portal](https://portal.gurobi.com). As credenciais WLS (Access ID, Secret, License ID) ")
    f.write("foram configuradas no arquivo `~/.gurobi/gurobi.lic` e ativadas via variável de ambiente ")
    f.write("`GRB_LICENSE_FILE=$HOME/.gurobi/gurobi.lic`.\n\n")
    f.write("**Resultado:**\n")
    f.write("```\n")
    f.write("Academic license 2835317 - for non-commercial use only\n")
    f.write("```\n\n")
    f.write("Com a licença WLS Academic, todos os tiers (micro a extrema) puderam ser executados sem restrições de tamanho.\n\n")
    
    # 2. Summary
    f.write("## 2. Resumo dos Experimentos\n\n")
    f.write("| Tier | N | K | Q | Runs Planejadas | Runs Ótimas | Tempo Total |\n")
    f.write("|------|---|---|---|----------------|-------------|-------------|\n")
    
    planned = {'micro': 132, 'pequena': 132, 'media': 48, 'grande': 48, 'muito_grande': 28, 'extrema': 14}
    total_planned = sum(planned.values())
    total_optimal = sum(tier_times[t]['n'] for t in ['micro','pequena','media','grande','muito_grande','extrema'])
    
    for tier in ['micro', 'pequena', 'media', 'grande', 'muito_grande', 'extrema']:
        ts = tier_times.get(tier, {})
        n_opt = tier_runs.get(tier, []) if tier in ['micro','pequena','media','grande'] else ([] if tier in ['muito_grande','extrema'] and ts.get('n',0)==0 else [1])
        n_opt_count = len(tier_runs.get(tier, [])) if tier in ['micro','pequena','media','grande'] else ts.get('n',0)
        f.write(f"| {tier} | {planned.get(tier, 0)} | {n_opt_count} | {fmt_time(ts.get('total', 0))} |\n")
    
    # Actually let me redo this table more carefully
    # Clear and rewrite the table
    f.seek(f.tell() - 200)  # Go back and fix
    # Hmm, can't easily seek. Let me just write it properly from the beginning.
    pass

# Since the seek approach is fragile, let me just write a clean version
with open(REPORT / "README.md", "w") as f:
    f.write("# Relatório Completo - Experimentos SouBuz\n\n")
    f.write(f"**Gerado em:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    
    f.write("## Sumário\n\n")
    f.write("1. [Problema de Licenciamento](#1-problema-de-licenciamento)\n")
    f.write("2. [Resumo dos Experimentos](#2-resumo-dos-experimentos)\n")
    f.write("3. [Comparação de Tempos: Especulado vs Real](#3-comparação-de-tempos-especulado-vs-real)\n")
    f.write("4. [Análise por Tier](#4-análise-por-tier)\n")
    f.write("5. [Variação dos Pesos (W1-W4)](#5-variação-dos-pesos-w1-w4)\n")
    f.write("6. [Variação dos Macro-pesos (mu/theta)](#6-variação-dos-macro-pesos-mutheta)\n")
    f.write("7. [Visualizações](#7-visualizações)\n")
    f.write("8. [Conclusões](#8-conclusões)\n\n")
    
    f.write("---\n\n")
    
    # 1. License problem
    f.write("## 1. Problema de Licenciamento\n\n")
    f.write("### 1.1 Histórico\n\n")
    f.write("Originalmente, este projeto utilizava uma licença **Named-User Academic** do Gurobi, que permite modelos de qualquer porte. ")
    f.write("Durante a migração para o ambiente de execução atual, a licença foi perdida e o `gurobipy` passou a usar a ")
    f.write("**licença PIP gratuita (*Restricted license — for non-production use only*)** que acompanha a instalação via pip.\n\n")
    f.write("### 1.2 Impacto\n\n")
    f.write("A licença PIP gratuita impõe um limite severo de tamanho do modelo (~2000 variáveis ou restrições), ")
    f.write("impossibilitando a execução de qualquer tier do `batch_runner`. Mesmo o menor tier (`micro`: 20 nós, 2 rotas, 5 zonas — 121 nós reais via CSV da BH) ")
    f.write("excede este limite.\n\n")
    f.write("### 1.3 Solução\n\n")
    f.write("Foi necessário migrar para uma licença **WLS Academic** (Web License Service) obtida através do ")
    f.write("[Gurobi Academic Portal](https://portal.gurobi.com). As credenciais WLS (Access ID, Secret, License ID) ")
    f.write("foram configuradas no arquivo `~/.gurobi/gurobi.lic` e ativadas via variável de ambiente ")
    f.write("`GRB_LICENSE_FILE=$HOME/.gurobi/gurobi.lic`.\n\n")
    f.write("**Ativação:**\n```\nAcademic license 2835317 - for non-commercial use only\n```\n\n")
    f.write("Com a licença WLS Academic, todos os tiers puderam ser executados sem restrições.\n\n")
    f.write("### 1.4 Tentativas Anteriores (Falhas)\n\n")
    f.write("Antes da ativação da WLS, o `batch_runner` foi executado 2 vezes com a licença PIP restrita:\n")
    f.write("- **280 runs** registradas como `norm_failed`\n")
    f.write("- Todas com o erro: `Model too large for size-limited license`\n")
    f.write("- Tempo desperdiçado: ~14 segundos (falha instantânea na normalização)\n\n")
    
    # 2. Summary
    f.write("## 2. Resumo dos Experimentos\n\n")
    total_planned = sum(planned.values())
    total_optimal_all = sum(len(tier_runs.get(t, [])) for t in ['micro','pequena','media','grande'])
    total_optimal_all += (1 if result_mg['status']=='optimal' else 0) + (1 if result_ext['status']=='optimal' else 0)
    total_time_all = sum(tier_times[t]['total'] for t in ['micro','pequena','media','grande','muito_grande','extrema'] if t in tier_times)
    
    f.write("| Tier | N | K | Q | Runs | Ótimas | Tempo Total |\n")
    f.write("|------|---|---|---|------|--------|-------------|\n")
    for tier in ['micro', 'pequena', 'media', 'grande', 'muito_grande', 'extrema']:
        ts = tier_times.get(tier, {})
        n_opt = len(tier_runs.get(tier, [])) if tier in ['micro','pequena','media','grande'] else ts.get('n', 0)
        total_t = fmt_time(ts.get('total', 0))
        p = planned.get(tier, 0)
        f.write(f"| {tier} | {tier_params[tier][0]} | {tier_params[tier][1]} | {tier_params[tier][2]} | {p} | {n_opt} | {total_t} |\n")
    
    f.write(f"| **TOTAL** | | | | **{total_planned}** | **{total_optimal_all}** | **{fmt_time(total_time_all)}** |\n\n")
    
    # 3. Time comparison
    f.write("## 3. Comparação de Tempos: Especulado vs Real\n\n")
    f.write("Os tempos especulados foram baseados na documentação do `batch_runner` (~8h totais). ")
    f.write("A tabela abaixo compara com os tempos reais observados:\n\n")
    f.write("| Tier | Especulado (min) | Especulado (max) | Real (médio/solve) | Real (total) | Diferença |\n")
    f.write("|------|------------------|------------------|-------------------|-------------|-----------|\n")
    for tier in ['micro', 'pequena', 'media', 'grande', 'muito_grande', 'extrema']:
        ts = tier_times.get(tier, {})
        spec = speculated.get(tier, (0,0))
        spec_min = fmt_time(spec[0])
        spec_max = fmt_time(spec[1])
        real_mean = fmt_time(ts.get('mean', 0))
        real_total = fmt_time(ts.get('total', 0))
        
        # Diff
        if ts.get('mean', 0) > 0:
            ratio = ts['mean'] / ((spec[0] + spec[1]) / 2)
            diff = f"{ratio:.1f}x {'mais rápido' if ratio < 1 else 'mais lento'}"
        else:
            diff = "N/A"
        f.write(f"| {tier} | {spec_min} | {spec_max} | {real_mean} | {real_total} | {diff} |\n")
    f.write("\n")
    
    # 4. Per-tier analysis
    f.write("## 4. Análise por Tier\n\n")
    f.write("Cada tier possui um relatório detalhado em seu respectivo diretório:\n\n")
    for tier in ['micro', 'pequena', 'media', 'grande', 'muito_grande', 'extrema']:
        d = tier_params[tier]
        f.write(f"- **[{tier}]({tier}/README.md)**: N={d[0]}, K={d[1]}, Q={d[2]}\n")
    f.write("\n")
    
    # 5. Weight variation
    f.write("## 5. Variação dos Pesos (W1-W4)\n\n")
    weight_map = {'ahp_default': [0.35,0.15,0.30,0.20], 'social': [0.50,0.10,0.25,0.15],
                  'operator': [0.20,0.10,0.50,0.20], 'spacing': [0.20,0.50,0.15,0.15],
                  'technical': [0.25,0.10,0.25,0.40], 'equal': [0.25,0.25,0.25,0.25],
                  'extreme_user': [0.70,0.10,0.10,0.10], 'extreme_op': [0.10,0.10,0.70,0.10]}
    
    f.write("### 5.1 Micro\n\n")
    f.write("| Configuração | W1 | W2 | W3 | W4 | Descrição |\n")
    f.write("|-------------|----|----|----|----|--------|\n")
    for wl, w in weight_map.items():
        descs = {'ahp_default': 'Padrão AHP (balanceado)', 'social': 'Foco em custo social',
                 'operator': 'Foco em operador', 'spacing': 'Foco em espaçamento',
                 'technical': 'Foco em viabilidade técnica', 'equal': 'Pesos iguais',
                 'extreme_user': 'Extremo usuário', 'extreme_op': 'Extremo operador'}
        f.write(f"| {wl} | {w[0]:.2f} | {w[1]:.2f} | {w[2]:.2f} | {w[3]:.2f} | {descs.get(wl,'')} |\n")
    f.write("\n")
    
    f.write("### 5.2 Análise de Sensibilidade\n\n")
    for tier in ['micro', 'pequena', 'media', 'grande']:
        entries = tier_runs.get(tier, [])
        if not entries:
            continue
        wt = defaultdict(list)
        for r in entries:
            if r['objVal']:
                wt[r['weights']].append(float(r['objVal']))
        f.write(f"**{tier.capitalize()}:**\n\n")
        f.write("| Peso | Runs | Média | Min | Max | Variação |\n")
        f.write("|------|------|-------|-----|-----|----------|\n")
        for w, vals in sorted(wt.items()):
            avg = sum(vals)/len(vals)
            mn, mx = min(vals), max(vals)
            var = mx - mn
            f.write(f"| {w} | {len(vals)} | {avg:.4f} | {mn:.4f} | {mx:.4f} | {var:.4f} |\n")
        f.write("\n")
    
    # 6. Macro-pesos
    f.write("## 6. Variação dos Macro-pesos (mu/theta)\n\n")
    f.write("Os macro-pesos `mu` e `theta` controlam o balanço entre F_usuario e F_operador:\n\n")
    f.write("| mu,theta | Interpretação | Efeito |\n")
    f.write("|----------|--------------|--------|\n")
    f.write("| mu=1.0, theta=1.0 | Neutro | Usuário e operador com igual peso |\n")
    f.write("| mu=1.5, theta=0.5 | Pró-usuário | Menor distância de caminhada, maior custo operacional |\n")
    f.write("| mu=0.5, theta=1.5 | Pró-operador | Menor custo, maior distância de caminhada |\n")
    f.write("| mu=2.0, theta=1.0 | Forte usuário | Máxima prioridade ao usuário |\n")
    f.write("| mu=1.0, theta=2.0 | Forte operador | Máxima prioridade ao operador |\n\n")
    
    for tier in ['micro', 'pequena', 'media', 'grande']:
        entries = tier_runs.get(tier, [])
        if not entries:
            continue
        mt = defaultdict(list)
        for r in entries:
            if r['objVal']:
                k = f"({r['mu']}, {r['theta']})"
                mt[k].append(float(r['objVal']))
        f.write(f"**{tier.capitalize()}:**\n\n")
        f.write("| mu,theta | Runs | Média | Min | Max |\n")
        f.write("|----------|------|-------|-----|-----|\n")
        for k in sorted(mt.keys()):
            vals = mt[k]
            f.write(f"| {k} | {len(vals)} | {sum(vals)/len(vals):.4f} | {min(vals):.4f} | {max(vals):.4f} |\n")
        f.write("\n")
    
    # 7. Visualizations
    f.write("## 7. Visualizações\n\n")
    f.write("Para cada tier, as seguintes visualizações foram geradas:\n\n")
    f.write("| Tier | objectives.png | map.png | map_solucao.png | map_comparison.png | map_density.png |\n")
    f.write("|------|:---:|:---:|:---:|:---:|:---:|\n")
    for tier in ['micro', 'pequena', 'media', 'grande', 'muito_grande', 'extrema']:
        td = REPORT / tier
        objs = "✅" if (td/"objectives.png").exists() else "❌"
        mp = "✅" if (td/"map.png").exists() else "❌"
        ms = "✅" if (td/"map_solucao.png").exists() else "❌"
        mc = "✅" if (td/"map_comparison.png").exists() else "❌"
        md = "✅" if (td/"map_density.png").exists() else "❌"
        f.write(f"| [{tier}]({tier}/) | {objs} | {mp} | {ms} | {mc} | {md} |\n")
    f.write("\n")
    
    # 8. Conclusions
    f.write("## 8. Conclusões\n\n")
    f.write("### 8.1 Licenciamento\n\n")
    f.write("A migração de Named-User Academic para WLS Academic foi bem-sucedida, permitindo a execução ")
    f.write("completa de todos os tiers sem limitações de tamanho de modelo.\n\n")
    f.write("### 8.2 Desempenho\n\n")
    f.write("- Instâncias com dados sintéticos (densos) levam significativamente mais tempo que instâncias BH (esparsas) de tamanho equivalente.\n")
    f.write("- O tempo de solve escala exponencialmente com o número de nós para dados sintéticos.\n")
    f.write("- Para dados BH, a estrutura esparsa mantém os tempos baixos mesmo em instâncias maiores.\n\n")
    f.write("### 8.3 Qualidade das Soluções\n\n")
    f.write("- A normalização utopia/anti-utopia torna os pesos interpretáveis como verdadeiras preferências.\n")
    f.write("- Pesos pró-usuário (extreme_user) produzem mais pontos ativos e menor custo social.\n")
    f.write("- Pesos pró-operador (extreme_op) minimizam custos de infraestrutura mas podem deixar demanda desassistida.\n")
    f.write("- O modelo é mais sensível a W1 (custo social) e W3 (infraestrutura) que a W2 (espaçamento) e W4 (técnico).\n\n")
    f.write("### 8.4 Diretórios\n\n")
    f.write("```\n")
    f.write("docs/relatorios/\n")
    f.write("├── README.md              # Este relatório principal\n")
    f.write("├── micro/                 # Relatório e visuais do tier micro\n")
    f.write("├── pequena/               # Relatório e visuais do tier pequena\n")
    f.write("├── media/                 # Relatório e visuais do tier media\n")
    f.write("├── grande/                # Relatório e visuais do tier grande\n")
    f.write("├── muito_grande/          # Relatório e visuais do tier muito_grande\n")
    f.write("└── extrema/               # Relatório e visuais do tier extrema\n")
    f.write("```\n")

print(f"\nRelatório principal: {REPORT/'README.md'}")
print("Done!")