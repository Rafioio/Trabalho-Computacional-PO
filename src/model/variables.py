import gurobipy as gp
from gurobipy import GRB


def create_variables(model, data, domains):
    N = data["N"]
    K = data["K"]
    I = data["I"]
    a_domain = domains["a_domain"]
    S_Indices = domains["S_Indices"]

    vars = {}

    vars["x"] = model.addVars(N, vtype=GRB.BINARY, name="x")
    vars["x_k"] = model.addVars(I, vtype=GRB.BINARY, name="x_k")
    vars["a"] = model.addVars(a_domain, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="a")
    vars["Cap"] = model.addVars(K, vtype=GRB.INTEGER, lb=0, name="Cap")
    vars["Cad"] = model.addVar(vtype=GRB.INTEGER, lb=0, name="Cad")
    vars["s_k"] = model.addVars(S_Indices, vtype=GRB.CONTINUOUS, lb=0, name="s_k")

    return vars