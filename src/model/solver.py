import gurobipy as gp
from gurobipy import GRB
import json
from pathlib import Path

# Importações relativas corretas
from src.model.domains import build_a_domain, build_S_indices
from src.model.variables import create_variables
from src.model.objective import build_objective, build_objective_exprs
from src.model.constraints import (
    add_c1, add_c2, add_c3, add_c4, 
    add_c5, add_c6, add_c7, add_c8
)


def build_model(data, domains=None):
    """
    Build Gurobi model from data.
    
    Args:
        data: Dictionary with model data
        domains: Pre-computed domains (optional)
        
    Returns:
        Tuple of (model, variables, domains, objective_expressions)
    """
    if domains is None:
        domains = {
            "a_domain": build_a_domain(data),
            "S_Indices": build_S_indices(data),
        }

    model = gp.Model("soubuz")
    model.Params.OutputFlag = 0

    vars_dict = create_variables(model, data, domains)
    obj_exprs = build_objective_exprs(data, vars_dict, domains)
    
    # Add all constraints
    add_c1(model, data, vars_dict)
    add_c2(model, data, domains, vars_dict)
    add_c3(model, data, domains, vars_dict)
    add_c4(model, data, vars_dict)
    add_c5(model, data, domains, vars_dict)
    add_c6(model, data, domains, vars_dict)
    add_c7(model, data, vars_dict)
    add_c8(model, data, vars_dict)

    model.update()
    return model, vars_dict, domains, obj_exprs


def build_and_solve(data, verbose=True):
    """
    Build and solve the optimization model.
    
    Args:
        data: Dictionary with model data
        verbose: Whether to show Gurobi output
        
    Returns:
        Dictionary with results
    """
    model, vars_dict, domains, obj_exprs = build_model(data)
    build_objective(model, data, vars_dict, domains)
    
    if verbose:
        model.Params.OutputFlag = 1
    else:
        model.Params.OutputFlag = 0
    
    model.optimize()

    results = {
        "model": model, 
        "vars": vars_dict, 
        "obj_exprs": obj_exprs,
        "domains": domains
    }
    
    if model.status == GRB.OPTIMAL:
        results["status"] = "optimal"
        results["obj"] = model.objVal
        
        # Extract solution for easy access
        results["solution"] = {
            "x": {n: vars_dict["x"][n].X for n in data["N"]},
            "x_k": {(n, k): vars_dict["x_k"][n, k].X for (n, k) in data["I"]},
            "Cap": {k: vars_dict["Cap"][k].X for k in data["K"]},
            "Cad": vars_dict["Cad"].X,
            "pontos_ativos": [n for n in data["N"] if vars_dict["x"][n].X > 0.5],
            "f1": obj_exprs["f1"].getValue(),
            "f2": obj_exprs["f2"].getValue(),
            "f3": obj_exprs["f3"].getValue(),
            "f4": obj_exprs["f4"].getValue(),
        }
        
    elif model.status == GRB.INFEASIBLE:
        results["status"] = "infeasible"
        print("Model is infeasible. Computing IIS...")
        model.computeIIS()
        model.write("infeasible.ilp")
        print("IIS written to infeasible.ilp")
        
    else:
        results["status"] = f"status_{model.status}"
        print(f"Optimization ended with status: {model.status}")

    return results


def save_solution(results, output_path="solucao.json"):
    """
    Save solution to JSON file.
    
    Args:
        results: Results dictionary from build_and_solve
        output_path: Path to save solution
    """
    if results.get("status") != "optimal":
        print(f"Cannot save solution: status = {results['status']}")
        return
    
    solution = results.get("solution", {})
    
    output = {
        "status": "optimal",
        "valor_objetivo": results["obj"],
        "f1": solution.get("f1", 0),
        "f2": solution.get("f2", 0),
        "f3": solution.get("f3", 0),
        "f4": solution.get("f4", 0),
        "Cad": float(solution.get("Cad", 0)),
        "Cap": {str(k): float(v) for k, v in solution.get("Cap", {}).items()},
        "pontos_ativos": solution.get("pontos_ativos", []),
        "x": {str(n): float(v) for n, v in solution.get("x", {}).items()},
        "x_k": {f"({n},{k})": float(v) for (n, k), v in solution.get("x_k", {}).items()},
    }
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"Solution saved to: {output_path}")


# For backward compatibility
def solve_from_file(filepath, verbose=True):
    """
    Load data from file and solve the model.
    
    Args:
        filepath: Path to JSON or DAT file
        verbose: Whether to show Gurobi output
        
    Returns:
        Dictionary with results
    """
    from src.data.loader import load_data
    
    print(f"\nLoading data from: {filepath}")
    data = load_data(filepath)
    
    print(f"Data loaded: {data['NumN']} nodes, {data['NumK']} routes, {data['NumQ']} demand zones")
    print(f"Total demand: {sum(data['de']):.0f} passengers")
    
    return build_and_solve(data, verbose)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "../data/dados_generated.json"
    
    results = solve_from_file(filepath, verbose=True)
    
    if results["status"] == "optimal":
        print(f"\n{'='*60}")
        print(f"SOLUTION FOUND!")
        print(f"{'='*60}")
        print(f"Objective value: {results['obj']:.4f}")
        print(f"  - f1 (social cost): {results['solution']['f1']:.4f}")
        print(f"  - f2 (technical feasibility): {results['solution']['f2']:.4f}")
        print(f"  - f3 (infrastructure cost): {results['solution']['f3']:.4f}")
        print(f"  - f4 (spacing penalty): {results['solution']['f4']:.4f}")
        print(f"\nActive stops: {len(results['solution']['pontos_ativos'])}")
        print(f"Additional capacity (Cad): {results['solution']['Cad']:.0f}")
        
        save_solution(results)