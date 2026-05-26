#!/usr/bin/env python3
"""Entry point: load data, validate, build model, solve, print results."""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="SouBuz — bus stop optimization")
    parser.add_argument("--data", default=None, help="Path to .dat or .json file")
    parser.add_argument("--json", action="store_true", help="Load from JSON (faster)")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation")
    parser.add_argument("--normalize-weights", action="store_true", help="Normalize weights via utopia/anti-utopia payoff table")
    parser.add_argument("--verbose-normalization", action="store_true", help="Print payoff table and normalization details")
    return parser.parse_args()


def _normalize(data):
    if isinstance(data.get("I"), list):
        if data["I"] and isinstance(data["I"][0], dict):
            data["I"] = [(x["n"], x["k"]) for x in data["I"]]
        elif data["I"] and isinstance(data["I"][0], (list, tuple)):
            data["I"] = [(x[0], x[1]) for x in data["I"]]
    if isinstance(data.get("L"), list):
        if data["L"] and isinstance(data["L"][0], dict):
            data["L"] = [(x["q"], x["k"]) for x in data["L"]]
        elif data["L"] and isinstance(data["L"][0], (list, tuple)):
            data["L"] = [(x[0], x[1]) for x in data["L"]]
    return data


def main():
    args = parse_args()

    # ---- Load data ----
    if args.data and args.json:
        from src.data.loader import load_json
        data = _normalize(load_json(args.data))
    elif args.data:
        from src.data.loader import load_dat
        data = load_dat(args.data)
    else:
        from src.data.loader import load_dat, load_json, _JSON_PATH
        if os.path.exists(_JSON_PATH):
            data = _normalize(load_json())
        else:
            data = load_dat()

    # ---- Validate ----
    if not args.no_validate:
        from src.utils.validator import validate
        ok = validate(data)
        if not ok:
            print("Validation FAILED. Aborting.")
            sys.exit(1)
        print("Validation OK.")

    # ---- Weight normalization ----
    if args.normalize_weights:
        from src.utils.weight_normalizer import normalize_weights
        print("Normalizando pesos via matriz payoff utopia/anti-utopia...")
        normalize_weights(data, verbose=args.verbose_normalization)
        fac = data["_normalization"]["factors"]
        print(f"Fatores de normalização: f1={fac[0]:.6f}  f2={fac[1]:.6f}  f3={fac[2]:.6f}  f4={fac[3]:.6f}")
        print(f"Pós-normalização: W1={data['W1']:.6f}  W2={data['W2']:.6f}  W3={data['W3']:.6f}  W4={data['W4']:.6f}")

    # ---- Solve ----
    from src.model.solver import build_and_solve
    from gurobipy import GRB

    results = build_and_solve(data, verbose=True)

    model = results["model"]

    if results["status"] == "optimal":
        print(f"\n{'='*50}")
        print(f"Valor da função objetivo: {results['obj']:.4f}")
        print(f"{'='*50}")

        obj_exprs = results["obj_exprs"]
        print()
        print(f"f1 (custo social):            {obj_exprs['f1'].getValue():.2f}")
        print(f"f2 (viabilidade técnica):     {obj_exprs['f2'].getValue():.2f}")
        print(f"f3 (custo de infraestrutura): {obj_exprs['f3'].getValue():.2f}")
        print(f"f4 (penalidade de espaçamento):{obj_exprs['f4'].getValue():.2f}")

        vars = results["vars"]
        print("\nCapacidade das rotas:")
        for k in data["K"]:
            print(f"  Rota {k}: Capacidade = {int(vars['Cap'][k].X)}")
        print(f"\nCapacidade adicional (Cad): {int(vars['Cad'].X)}")
    else:
        print(f"Status: {results['status']}")


if __name__ == "__main__":
    main()