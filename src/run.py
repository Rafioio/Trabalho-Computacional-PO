#!/usr/bin/env python3
"""
Entry point: load data, validate, build model, solve, print results.

Usage:
    python -m src.run                                    # Auto-detect data file
    python -m src.run --data dados_generated.json        # Load specific JSON
    python -m src.run --data dados_generated.json --json # Force JSON loader
    python -m src.run --data dados.dat --normalize-weights # With weight normalization
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="SouBuz — bus stop optimization")
    parser.add_argument("--data", "-d", default=None, help="Path to .dat or .json file")
    parser.add_argument("--json", action="store_true", help="Force JSON loader (auto-detected by default)")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation")
    parser.add_argument("--normalize-weights", action="store_true", 
                       help="Normalize weights via utopia/anti-utopia payoff table")
    parser.add_argument("--verbose-normalization", action="store_true", 
                       help="Print payoff table and normalization details")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Show Gurobi solver output")
    parser.add_argument("--output", "-o", default="solucao.json",
                       help="Output solution file (default: solucao.json)")
    return parser.parse_args()


def normalize_data_structure(data):
    """
    Normalize data structure to expected format.
    Handles differences between DAT and JSON formats.
    """
    # Normalize I (pairs of n,k)
    if isinstance(data.get("I"), list):
        if data["I"] and isinstance(data["I"][0], dict):
            data["I"] = [(x["n"], x["k"]) for x in data["I"]]
        elif data["I"] and isinstance(data["I"][0], (list, tuple)):
            if len(data["I"][0]) == 2:
                data["I"] = [(x[0], x[1]) for x in data["I"]]
    
    # Normalize L (pairs of q,k)
    if isinstance(data.get("L"), list):
        if data["L"] and isinstance(data["L"][0], dict):
            data["L"] = [(x["q"], x["k"]) for x in data["L"]]
        elif data["L"] and isinstance(data["L"][0], (list, tuple)):
            if len(data["L"][0]) == 2:
                data["L"] = [(x[0], x[1]) for x in data["L"]]
    
    # Ensure derived sets exist
    if "Q" not in data and "NumQ" in data:
        data["Q"] = list(range(1, data["NumQ"] + 1))
    if "N" not in data and "NumN" in data:
        data["N"] = list(range(1, data["NumN"] + 1))
    if "K" not in data and "NumK" in data:
        data["K"] = list(range(1, data["NumK"] + 1))
    
    # Ensure T (inactive nodes) exists
    if "T" not in data and "C" in data and "N" in data:
        data["T"] = [n for n in data["N"] if n not in data["C"]]
    
    return data


def validate_data(data):
    """
    Quick validation of required data fields.
    Returns True if data is valid, False otherwise.
    """
    required_fields = ["NumN", "NumK", "NumQ", "de", "w", "d", "I", "L"]
    
    missing = [f for f in required_fields if f not in data]
    if missing:
        print(f"Missing required fields: {missing}")
        return False
    
    # Check dimensions
    if len(data["de"]) != data["NumQ"]:
        print(f"Warning: de length ({len(data['de'])}) != NumQ ({data['NumQ']})")
        # Pad or truncate
        if len(data["de"]) < data["NumQ"]:
            data["de"].extend([50] * (data["NumQ"] - len(data["de"])))
        else:
            data["de"] = data["de"][:data["NumQ"]]
    
    if len(data["w"]) != data["NumN"]:
        print(f"Warning: w length ({len(data['w'])}) != NumN ({data['NumN']})")
        if len(data["w"]) < data["NumN"]:
            data["w"].extend([0.5] * (data["NumN"] - len(data["w"])))
        else:
            data["w"] = data["w"][:data["NumN"]]
    
    # Check I set (non-empty)
    if not data["I"]:
        print("Warning: I set is empty! Creating default I from V if available.")
        if "V" in data and data["V"]:
            data["I"] = []
            for k_idx, route in enumerate(data["V"]):
                k = k_idx + 1
                for n in route:
                    data["I"].append((n, k))
    
    if not data["L"]:
        print("Warning: L set is empty! Creating default L (each demand served by first 3 routes).")
        data["L"] = []
        for q in data["Q"]:
            for k in range(1, min(4, data["NumK"] + 1)):
                data["L"].append((q, k))
    
    return True


def load_data_file(filepath, force_json=False):
    """
    Load data from file with automatic format detection.
    """
    from src.data.loader import load_dat, load_json, load_data
    
    if filepath:
        # Force JSON loader if requested
        if force_json or Path(filepath).suffix.lower() == '.json':
            print(f"Loading JSON file: {filepath}")
            return load_json(filepath)
        elif Path(filepath).suffix.lower() == '.dat':
            print(f"Loading DAT file: {filepath}")
            return load_dat(filepath)
        else:
            # Auto-detect
            print(f"Auto-detecting format for: {filepath}")
            return load_data(filepath)
    else:
        # No file specified, try default locations
        from src.data.loader import _JSON_PATH, _DAT_PATH
        
        if os.path.exists(_JSON_PATH):
            print(f"Loading default JSON: {_JSON_PATH}")
            return load_json(_JSON_PATH)
        elif os.path.exists(_DAT_PATH):
            print(f"Loading default DAT: {_DAT_PATH}")
            return load_dat(_DAT_PATH)
        else:
            raise FileNotFoundError(f"No data file found. Expected {_JSON_PATH} or {_DAT_PATH}")


def save_solution_to_file(results, output_path, data):
    """
    Save solution to JSON file for visualization.
    """
    if results.get("status") != "optimal":
        print(f"Cannot save solution: status = {results['status']}")
        return
    
    import json
    
    obj_exprs = results.get("obj_exprs", {})
    vars_dict = results.get("vars", {})
    
    solution = {
        "status": "optimal",
        "valor_objetivo": results["obj"],
        "f1": obj_exprs.get("f1", {}).getValue() if hasattr(obj_exprs.get("f1", {}), "getValue") else 0,
        "f2": obj_exprs.get("f2", {}).getValue() if hasattr(obj_exprs.get("f2", {}), "getValue") else 0,
        "f3": obj_exprs.get("f3", {}).getValue() if hasattr(obj_exprs.get("f3", {}), "getValue") else 0,
        "f4": obj_exprs.get("f4", {}).getValue() if hasattr(obj_exprs.get("f4", {}), "getValue") else 0,
        "Cad": float(vars_dict.get("Cad", {}).X) if hasattr(vars_dict.get("Cad", {}), "X") else 0,
        "Cap": {str(k): float(vars_dict["Cap"][k].X) for k in data.get("K", []) if k in vars_dict.get("Cap", {})},
        "pontos_ativos": [n for n in data.get("N", []) 
                         if n in vars_dict.get("x", {}) and vars_dict["x"][n].X > 0.5],
    }
    
    # Add x_k values for active route-stops
    solution["x_k_ativos"] = [
        {"n": n, "k": k} for (n, k) in data.get("I", [])
        if (n, k) in vars_dict.get("x_k", {}) and vars_dict["x_k"][n, k].X > 0.5
    ]
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(solution, f, indent=2, ensure_ascii=False)
    
    print(f"\nSolution saved to: {output_path}")


def print_summary(data, results):
    """
    Print detailed summary of results.
    """
    if results["status"] != "optimal":
        print(f"\nOptimization ended with status: {results['status']}")
        return
    
    print("\n" + "=" * 60)
    print("SOUBUZ - OPTIMAL SOLUTION FOUND")
    print("=" * 60)
    
    print(f"\n📊 Objective value: {results['obj']:.4f}")
    print("\n📈 Objective components:")
    print(f"   f₁ (custo social):            {results['obj_exprs']['f1'].getValue():.2f}")
    print(f"   f₂ (viabilidade técnica):     {results['obj_exprs']['f2'].getValue():.2f}")
    print(f"   f₃ (custo infraestrutura):    {results['obj_exprs']['f3'].getValue():.2f}")
    print(f"   f₄ (penalidade espaçamento):  {results['obj_exprs']['f4'].getValue():.2f}")
    
    print("\n🚌 Route capacities:")
    for k in data["K"]:
        cap = int(results['vars']["Cap"][k].X)
        print(f"   Rota {k}: {cap} passengers")
    
    print(f"\n💰 Additional capacity (Cad): {int(results['vars']['Cad'].X)}")
    
    # Active stops
    active_stops = [n for n in data["N"] if results['vars']["x"][n].X > 0.5]
    print(f"\n📍 Active stops: {len(active_stops)}/{data['NumN']}")
    if active_stops:
        print(f"   IDs: {active_stops[:20]}")
        if len(active_stops) > 20:
            print(f"   ... and {len(active_stops) - 20} more")
    
    # Active route-stops
    active_route_stops = [(n, k) for (n, k) in data["I"] 
                          if results['vars']["x_k"][n, k].X > 0.5]
    print(f"\n🚏 Active route-stops: {len(active_route_stops)}")
    
    # Utilization statistics
    total_capacity = sum(results['vars']["Cap"][k].X for k in data["K"])
    total_demand = sum(data["de"])
    utilization = (total_demand / total_capacity * 100) if total_capacity > 0 else 0
    print(f"\n📊 System utilization: {utilization:.1f}% (demand {total_demand:.0f} / capacity {total_capacity:.0f})")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("SOUBUZ — BUS STOP OPTIMIZATION MODEL")
    print("=" * 60)
    
    # ---- Load data ----
    print("\n[1/4] Loading data...")
    try:
        data = load_data_file(args.data, force_json=args.json)
        data = normalize_data_structure(data)
        print(f"✓ Loaded: {data['NumN']} nodes, {data['NumK']} routes, {data['NumQ']} demand zones")
        print(f"  Total demand: {sum(data['de']):.0f} passengers")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        sys.exit(1)
    
    # ---- Validate ----
    print("\n[2/4] Validating data...")
    if not args.no_validate:
        try:
            from src.utils.validator import validate
            ok = validate(data)
            if not ok:
                print("✗ Validation FAILED. Aborting.")
                sys.exit(1)
            print("✓ Validation OK.")
        except ImportError:
            print("⚠ Validator module not found, using internal validation...")
            if not validate_data(data):
                print("✗ Validation FAILED. Aborting.")
                sys.exit(1)
            print("✓ Validation OK.")
    else:
        print("⚠ Validation skipped.")
    
    # ---- Weight normalization ----
    print("\n[3/4] Setting up objective function...")
    if args.normalize_weights:
        try:
            from src.utils.weight_normalizer import normalize_weights
            print("Normalizing weights via utopia/anti-utopia payoff table...")
            normalize_weights(data, verbose=args.verbose_normalization)
            fac = data["_normalization"]["factors"]
            print(f"✓ Normalization factors: f1={fac[0]:.6f}  f2={fac[1]:.6f}  f3={fac[2]:.6f}  f4={fac[3]:.6f}")
            print(f"  Post-normalization weights: W1={data['W1']:.6f}  W2={data['W2']:.6f}  W3={data['W3']:.6f}  W4={data['W4']:.6f}")
        except ImportError:
            print("⚠ Weight normalizer module not found, using raw weights.")
    else:
        print(f"Using raw weights: W1={data['W1']:.2f}, W2={data['W2']:.2f}, W3={data['W3']:.2f}, W4={data['W4']:.2f}")
    
    # ---- Solve ----
    print("\n[4/4] Solving optimization model...")
    print("-" * 40)
    
    from src.model.solver import build_and_solve
    
    # Set verbose based on args
    results = build_and_solve(data, verbose=args.verbose)
    
    # ---- Results ----
    if results["status"] == "optimal":
        print_summary(data, results)
        
        # Save solution for visualization
        save_solution_to_file(results, args.output, data)
        
    elif results["status"] == "infeasible":
        print("\n" + "=" * 60)
        print("⚠ MODEL IS INFEASIBLE!")
        print("=" * 60)
        print("\nPossible causes:")
        print("  • Inconsistent route definitions (V matrix)")
        print("  • Terminal constraints (c4) that cannot be satisfied")
        print("  • Capacity constraints too tight (Capt too low)")
        print("  • Spacing constraints (c5) with d_route_max too small")
        print("\nIIS written to 'infeasible.ilp'")
        print("Analyze with: gurobi_cl infeasible.ilp")
        
    else:
        print(f"\n⚠ Optimization ended with status: {results['status']}")
    
    print("\n" + "=" * 60)
    print("SOUBUZ FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()