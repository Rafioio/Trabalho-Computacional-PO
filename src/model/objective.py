import gurobipy as gp
from gurobipy import GRB


def build_objective(model, data, vars, domains):
    Q = data["Q"]
    N = data["N"]
    K = data["K"]
    I = data["I"]
    T = data["T"]
    de = data["de"]
    d = data["d"]
    w = data["w"]
    omega = data["omega"]
    P = data["P"]
    W1 = data["W1"]
    W2 = data["W2"]
    W3 = data["W3"]
    W4 = data["W4"]

    a_domain = domains["a_domain"]
    S_Indices = domains["S_Indices"]

    f1 = gp.quicksum(
        de[q - 1]
        * (
            gp.quicksum(
                d[q - 1][n - 1] * vars["a"][q, n, k]
                for (qq, n, k) in a_domain
                if qq == q
            )
            + (
                1
                - gp.quicksum(
                    vars["a"][q, n, k]
                    for (qq, n, k) in a_domain
                    if qq == q
                )
            )
            * P
        )
        for q in Q
    )

    f2 = gp.quicksum((1 / w[n - 1]) * vars["x_k"][n, k] for (n, k) in I)

    f3 = omega * vars["Cad"] + gp.quicksum(vars["x"][n] for n in T)

    f4 = gp.quicksum(vars["s_k"][k, idx] for (k, idx) in S_Indices)

    model.setObjective(W1 * f1 + W2 * f2 + W3 * f3 + W4 * f4, GRB.MINIMIZE)
    return {"f1": f1, "f2": f2, "f3": f3, "f4": f4}