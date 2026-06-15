#!/usr/bin/env python3
"""
Batch experiment runner for SouBuz — executes 6 instance tiers across multiple
seeds and weight combinations, saving full results to docs/runs/.

Usage:
    python -m src.utils.batch_runner                    # Run all tiers
    python -m src.utils.batch_runner --tiers micro,media # Selected tiers
    python -m src.utils.batch_runner --dry-run           # Print plan only
    python -m src.utils.batch_runner --quick             # Quarter time (debug)
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import gurobipy as gp
import numpy as np

from src.model.objective import build_objective
from src.model.solver import build_model

# ============================================================================
# Tier definition
# ============================================================================
TIERS = []

WEIGHT_GRID = {
    "full": [
        ("ahp_default",   [0.35, 0.15, 0.30, 0.20]),
        ("social",        [0.50, 0.10, 0.25, 0.15]),
        ("operator",      [0.20, 0.10, 0.50, 0.20]),
        ("spacing",       [0.20, 0.50, 0.15, 0.15]),
        ("technical",     [0.25, 0.10, 0.25, 0.40]),
        ("equal",         [0.25, 0.25, 0.25, 0.25]),
        ("extreme_user",  [0.70, 0.10, 0.10, 0.10]),
        ("extreme_op",    [0.10, 0.10, 0.70, 0.10]),
    ],
    "medium": [
        ("ahp_default",   [0.35, 0.15, 0.30, 0.20]),
        ("social",        [0.50, 0.10, 0.25, 0.15]),
        ("operator",      [0.20, 0.10, 0.50, 0.20]),
        ("equal",         [0.25, 0.25, 0.25, 0.25]),
    ],
    "sparse": [
        ("ahp_default",   [0.35, 0.15, 0.30, 0.20]),
        ("operator",      [0.20, 0.10, 0.50, 0.20]),
    ],
}


def _define_tiers(quick=False):
    TIERS.clear()
    s = 0.5 if quick else 1.0
    TIERS.append(("micro",           int(20*s),  2, int(5*s),  3,  WEIGHT_GRID["full"],   60))
    TIERS.append(("pequena",         int(50*s),  4, int(10*s), 3,  WEIGHT_GRID["full"],  120))
    TIERS.append(("media",           int(150*s), 6, int(20*s), 2,  WEIGHT_GRID["medium"], 300))
    TIERS.append(("grande",          int(300*s), 8, int(30*s), 2,  WEIGHT_GRID["medium"], 600))
    TIERS.append(("muito_grande",    int(500*s),10, int(30*s), 2,  WEIGHT_GRID["sparse"],1200))
    TIERS.append(("extrema",         int(800*s),15, int(40*s), 1,  WEIGHT_GRID["sparse"],1800))
    return TIERS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="SouBuz batch experiment runner — 6 tiers, ~8 h runtime",
    )
    parser.add_argument("--tiers", default=None,
                        help="Comma-separated tier names to run (default: all)")
    parser.add_argument("--output-dir", default="docs/runs",
                        help="Output root (default: docs/runs)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print experiment plan and exit")
    parser.add_argument("--quick", action="store_true",
                        help="Half-size instances for debugging")
    return parser.parse_args(argv)


def generate_data_for_tier(tier_label, N, K, Q, seed):
    from src.utils.generate_data import generate_data, parse_args as gen_args
    args = gen_args([
        "--num-n", str(N),
        "--num-k", str(K),
        "--num-q", str(Q),
        "--seed", str(seed),
        "--output", "/dev/null",
        "--quiet",
    ])
    rng = np.random.default_rng(seed)
    data = generate_data(args, rng, seed)
    data["mu"] = 1.0
    data["theta"] = 1.0
    return data


def solve_run(data, W, time_limit, verbose=False):
    data["W1"], data["W2"], data["W3"], data["W4"] = W
    from src.model.domains import build_a_domain, build_S_indices
    from src.model.variables import create_variables
    from src.model.constraints import (
        add_c1, add_c2, add_c3, add_c4,
        add_c5, add_c6, add_c7, add_c8,
    )

    domains = {
        "a_domain": build_a_domain(data),
        "S_Indices": build_S_indices(data),
    }

    model = gp.Model("soubuz_batch")
    model.Params.OutputFlag = 0
    model.Params.NonConvex = 2
    model.Params.TimeLimit = time_limit

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

    if not verbose:
        model.Params.OutputFlag = 0
    else:
        model.Params.OutputFlag = 1

    model.optimize()

    status_map = {
        gp.GRB.OPTIMAL: "optimal",
        gp.GRB.INFEASIBLE: "infeasible",
        gp.GRB.TIME_LIMIT: "time_limit",
        gp.GRB.UNBOUNDED: "unbounded",
        gp.GRB.INF_OR_UNBD: "inf_or_unbd",
    }
    status = status_map.get(model.status, f"status_{model.status}")

    result = {
        "status": status,
        "objVal": model.objVal if model.status == gp.GRB.OPTIMAL else None,
        "model": model,
        "vars": vars_dict,
        "obj_exprs": obj_exprs,
        "domains": domains,
    }

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

    return result


def _norm_val(val, lo, hi):
    d = hi - lo
    return (val - lo) / d if abs(d) > 1e-12 else 0.0


def save_run(run_dir, data, result, W, weight_label, tier_label,
             solve_time, norm_time, seed):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "tier": tier_label,
        "seed": seed,
        "NumN": data["NumN"],
        "NumK": data["NumK"],
        "NumQ": data["NumQ"],
        "W1": W[0], "W2": W[1], "W3": W[2], "W4": W[3],
        "weight_label": weight_label,
        "mu": data.get("mu", 1),
        "theta": data.get("theta", 1),
        "per_solve_limit_s": data.get("_time_limit", 0),
        "normalization_time_s": round(norm_time, 2),
        "solve_time_s": round(solve_time, 2),
        "timestamp": datetime.now().isoformat(),
    }
    with open(run_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    norm_data = data.get("_normalization", {})
    utopia = norm_data.get("utopia", [0, 0, 0, 0])
    anti = norm_data.get("anti_utopia", [1, 1, 1, 1])

    summary = {"status": result["status"], "objVal": result.get("objVal")}
    if result["status"] == "optimal":
        sol = result["solution"]
        summary["f1_raw"] = sol["f1"]
        summary["f2_raw"] = sol["f2"]
        summary["f3_raw"] = sol["f3"]
        summary["f4_raw"] = sol["f4"]
        summary["f1_norm"] = _norm_val(sol["f1"], utopia[0], anti[0])
        summary["f2_norm"] = _norm_val(sol["f2"], utopia[1], anti[1])
        summary["f3_norm"] = _norm_val(sol["f3"], utopia[2], anti[2])
        summary["f4_norm"] = _norm_val(sol["f4"], utopia[3], anti[3])
        summary["F_usuario"] = sol["F_usuario"]
        summary["F_operador"] = sol["F_operador"]
        summary["Cad"] = sol["Cad"]
        summary["total_buses"] = sum(sol["Cap"].values()) + sol["Cad"]
        summary["active_stops"] = len(sol["pontos_ativos"])
        summary["route_capacities"] = sol["Cap"]

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if result["status"] == "optimal":
        sol = result["solution"]
        solucao = {
            "valor_objetivo": round(result["objVal"], 4),
            "status": "optimal",
            "num_pontos_ativos": len(sol["pontos_ativos"]),
            "pontos_ativos": sol["pontos_ativos"],
            "capacidade_rotas": sol["Cap"],
            "capacidade_adicional": int(sol["Cad"]),
            "objetivos": {
                "f1_custo_social": round(sol["f1"], 2),
                "f2_penalidade_espacamento": round(sol["f2"], 2),
                "f3_custo_infraestrutura": round(sol["f3"], 2),
                "f4_viabilidade_tecnica": round(sol["f4"], 2),
                "F_usuario": round(sol["F_usuario"], 4),
                "F_operador": round(sol["F_operador"], 4),
            },
        }
        with open(run_dir / "solucao.json", "w") as f:
            json.dump(solucao, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if result["status"] == "optimal":
            sol = result["solution"]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            labels = ["f1 (social)", "f2 (spacing)", "f3 (infra)", "f4 (technical)"]
            raw_vals = [sol["f1"], sol["f2"], sol["f3"], sol["f4"]]
            bars1 = ax1.bar(labels, raw_vals,
                            color=["#2ecc71", "#e74c3c", "#3498db", "#f39c12"])
            ax1.set_title("Raw objective values")
            ax1.set_ylabel("Value")
            for bar, v in zip(bars1, raw_vals):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{v:.1f}", ha="center", va="bottom", fontsize=9)

            norm_vals = [
                _norm_val(sol["f1"], utopia[0], anti[0]),
                _norm_val(sol["f2"], utopia[1], anti[1]),
                _norm_val(sol["f3"], utopia[2], anti[2]),
                _norm_val(sol["f4"], utopia[3], anti[3]),
            ]
            weighted = [n * wi for n, wi in zip(norm_vals, W)]
            x = range(len(labels))
            ax2.bar(x, norm_vals, 0.6, label="Normalized", color="#95a5a6")
            ax2.bar(x, weighted, 0.6, label="Weighted", color="#e67e22", alpha=0.7)
            ax2.set_xticks(list(x))
            ax2.set_xticklabels(labels)
            ax2.set_title("Normalized vs Weighted objectives")
            ax2.set_ylabel("Normalized value [0,1]")
            ax2.legend()
            ax2.set_ylim(0, 1.15)

            fig.suptitle(
                f"{tier_label} | seed={seed} | {weight_label} "
                f"obj={result['objVal']:.4f}",
                fontsize=12, y=1.02,
            )
            plt.tight_layout()
            plt.savefig(run_dir / "objectives.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

        if result["status"] == "optimal" and "positions" in data:
            sol = result["solution"]
            positions = np.array(data["positions"])
            if len(positions) > 0:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.scatter(positions[:, 0], positions[:, 1],
                           c="#bdc3c7", s=15, alpha=0.4, label="Inactive")

                active = sol.get("pontos_ativos", [])
                for n in active:
                    idx = n - 1
                    if 0 <= idx < len(positions):
                        ax.scatter(positions[idx, 0], positions[idx, 1],
                                   c="#e74c3c", s=40, zorder=5)

                vars_dict = result.get("vars", {})
                route_stops_map = {}
                for n, k in data["I"]:
                    xk_var = vars_dict.get("x_k", {}).get((n, k))
                    if xk_var is not None and xk_var.X > 0.5:
                        route_stops_map.setdefault(k, []).append(n)

                colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                          "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                          "#bcbd22", "#17becf"]
                for kidx, (k, stops) in enumerate(route_stops_map.items()):
                    for n in stops:
                        idx = n - 1
                        if 0 <= idx < len(positions):
                            ax.scatter(positions[idx, 0], positions[idx, 1],
                                       color=colors[kidx % len(colors)],
                                       s=80, marker="s", edgecolors="black",
                                       linewidths=0.5, zorder=6)

                ax.set_title(f"{tier_label} | seed={seed} | {weight_label}")
                ax.set_aspect("equal")
                plt.tight_layout()
                plt.savefig(run_dir / "map.png", dpi=120, bbox_inches="tight")
                plt.close(fig)
    except ImportError:
        pass

    try:
        log_path = run_dir / "solver.log"
        if result.get("model") is not None:
            result["model"].write(str(log_path))
    except Exception:
        pass


def update_index(output_dir, tier_label, weight_label, seed, status,
                 objVal, solve_time, run_id, timestamp):
    index_path = Path(output_dir) / "runs_index.csv"
    exists = index_path.exists()
    with open(index_path, "a") as f:
        if not exists:
            f.write("run_id,tier,seed,weights,status,objVal,"
                    "solve_time_s,timestamp\n")
        f.write(
            f"{run_id},{tier_label},{seed},{weight_label},{status},"
            f"{objVal if objVal is not None else ''},"
            f"{solve_time:.2f},{timestamp}\n"
        )


def estimate_total_time(tiers):
    total_norm_solves = 0
    total_final_solves = 0
    total_est_s = 0

    for label, N, K, Q, n_seeds, weight_list, per_solve_limit in tiers:
        n_weights = len(weight_list)
        for _ in range(n_seeds):
            total_norm_solves += 4
            total_final_solves += n_weights
            est_norm = 4 * per_solve_limit * 0.3
            est_final = n_weights * per_solve_limit * 0.5
            total_est_s += est_norm + est_final

    return total_norm_solves, total_final_solves, total_est_s


def print_plan(tiers):
    total_runs = 0
    total_norms = 0
    for label, N, K, Q, n_seeds, weight_list, per_solve_limit in tiers:
        n_weights = len(weight_list)
        runs = n_seeds * n_weights
        norms = n_seeds * 4
        total_runs += runs
        total_norms += norms
        print(
            f"  {label:15s}  N={N:4d} K={K:2d} Q={Q:2d}  "
            f"seeds={n_seeds}  weights={n_weights:2d}  "
            f"limit={per_solve_limit:4d}s  "
            f"-> {runs:3d} final solves + {norms:2d} norm solves"
        )

    n, f, est = estimate_total_time(tiers)
    print(
        f"\nTotal: {n} normalization solves + {f} final solves "
        f"= {n + f} Gurobi solves"
    )
    print(f"Estimated: {timedelta(seconds=int(est))} (target: ~8 h)")


def run_tier(tier, base_dir, quick=False, verbose=False):
    label, N, K, Q, n_seeds, weight_list, per_solve_limit = tier

    for seed_idx in range(n_seeds):
        seed = 42 + seed_idx * 10
        print(f"\n{'='*60}")
        print(f"TIER: {label} | seed={seed} | N={N} K={K} Q={Q}")
        print(f"{'='*60}")

        t0 = time.time()
        data = generate_data_for_tier(label, N, K, Q, seed)
        gen_time = time.time() - t0
        data["_time_limit"] = per_solve_limit
        n_I = len(data.get("I", []))
        n_L = len(data.get("L", []))
        print(f"  Data generated in {gen_time:.1f}s (|I|={n_I}, |L|={n_L})")

        t1 = time.time()
        try:
            from src.model.function_normalizer import normalize_function
            normalize_function(data, verbose=verbose)
            norm_time = time.time() - t1
            print(f"  Normalization (4 solves) in {norm_time:.1f}s")
        except Exception as e:
            print(f"  NORMALIZATION FAILED: {e}")
            for wlabel, W in weight_list:
                run_id = _next_run_id(base_dir)
                run_dir = Path(base_dir) / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                result = {"status": "norm_failed", "objVal": None}
                save_run(run_dir, data, result, W, wlabel,
                         label, 0, 0, seed)
                update_index(base_dir, label, wlabel, seed,
                             "norm_failed", None, 0,
                             run_id, datetime.now().isoformat())
            return

        for wlabel, W in weight_list:
            t2 = time.time()
            try:
                result = solve_run(data, W, per_solve_limit, verbose=verbose)
            except Exception as e:
                print(f"  SOLVE FAILED ({wlabel}): {e}")
                result = {"status": "error", "objVal": None}
            solve_time = time.time() - t2

            run_id = _next_run_id(base_dir)
            run_dir = Path(base_dir) / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            save_run(run_dir, data, result, W, wlabel,
                     label, solve_time, norm_time, seed)
            update_index(base_dir, label, wlabel, seed,
                         result["status"], result.get("objVal"),
                         solve_time, run_id, datetime.now().isoformat())

            icons = {"optimal": "OK", "infeasible": "IN", "time_limit": "TL",
                     "error": "ER", "norm_failed": "NF"}
            icon = icons.get(result["status"], "??")
            obj_str = f"obj={result['objVal']:.4f}" if result.get("objVal") is not None else "no-obj"
            print(f"  [{icon}] {wlabel:15s} {result['status']:12s} "
                  f"{obj_str}  {solve_time:.1f}s  -> {run_id}")

        del data


def _next_run_id(base_dir):
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    nums = []
    for d in base.iterdir():
        if d.is_dir() and d.name.startswith("run_"):
            try:
                nums.append(int(d.name.split("_")[1]))
            except (IndexError, ValueError):
                pass
    next_num = max(nums) + 1 if nums else 1
    return f"run_{next_num:04d}"


def main():
    start_wall = time.time()
    args = parse_args()
    tiers = _define_tiers(quick=args.quick)

    if args.tiers:
        selected = set(args.tiers.split(","))
        tiers = [t for t in tiers if t[0] in selected]
        if not tiers:
            names = [t[0] for t in _define_tiers()]
            print(f"No matching tiers for: {args.tiers}")
            print(f"Available: {names}")
            sys.exit(1)

    base_dir = Path(args.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SOUBUZ BATCH EXPERIMENT RUNNER")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output dir: {base_dir.resolve()}")
    print("=" * 60)
    print("\nExperiment plan:")
    print_plan(tiers)

    if args.dry_run:
        return

    print(f"\n{'='*60}")
    print("BEGIN EXECUTION")
    print(f"{'='*60}\n")

    for tier in tiers:
        run_tier(tier, base_dir, quick=args.quick, verbose=False)

    wall = time.time() - start_wall

    grand_total = {"optimal": 0, "infeasible": 0, "time_limit": 0,
                   "error": 0, "norm_failed": 0}
    index_path = base_dir / "runs_index.csv"
    if index_path.exists():
        with open(index_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                st = row["status"]
                if st in grand_total:
                    grand_total[st] += 1

    total_runs = sum(grand_total.values())
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE — Wall time: {timedelta(seconds=int(wall))}")
    print(f"{'='*60}")
    print(f"Total runs executed: {total_runs}")
    for st, cnt in grand_total.items():
        if cnt > 0:
            print(f"  {st}: {cnt}")
    print(f"\nResults saved to: {base_dir.resolve()}")
    print(f"Index: {(base_dir / 'runs_index.csv').resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
