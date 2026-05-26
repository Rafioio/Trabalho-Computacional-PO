import gurobipy as gp
from gurobipy import GRB

from src.model.domains import build_a_domain, build_S_indices
from src.model.variables import create_variables
from src.model.objective import build_objective, build_objective_exprs
from src.model.constraints import add_c1, add_c2, add_c3, add_c4, add_c5, add_c6, add_c7, add_c8


def build_model(data, domains=None):
    if domains is None:
        domains = {
            "a_domain": build_a_domain(data),
            "S_Indices": build_S_indices(data),
        }

    model = gp.Model("soubuz")
    model.Params.OutputFlag = 0

    vars = create_variables(model, data, domains)
    obj_exprs = build_objective_exprs(data, vars, domains)
    add_c1(model, data, vars)
    add_c2(model, data, domains, vars)
    add_c3(model, data, domains, vars)
    add_c4(model, data, vars)
    add_c5(model, data, domains, vars)
    add_c6(model, data, domains, vars)
    add_c7(model, data, vars)
    add_c8(model, data, vars)

    return model, vars, domains, obj_exprs


def build_and_solve(data, verbose=True):
    model, vars, domains, obj_exprs = build_model(data)
    build_objective(model, data, vars, domains)
    if verbose:
        model.Params.OutputFlag = 1
    model.optimize()

    results = {"model": model, "vars": vars, "obj_exprs": obj_exprs}
    if model.status == GRB.OPTIMAL:
        results["status"] = "optimal"
        results["obj"] = model.objVal
    elif model.status == GRB.INFEASIBLE:
        results["status"] = "infeasible"
    else:
        results["status"] = f"status_{model.status}"

    return results