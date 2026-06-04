import gurobipy as gp
from gurobipy import GRB


def add_c1(model, data, vars):
    for n, k in data["I"]:
        model.addConstr(vars["x_k"][n, k] <= vars["x"][n],
                        name=f"c1_{n}_{k}")


def add_c2(model, data, domains, vars):
    for q, n, k in domains["a_domain"]:
        model.addConstr(vars["a"][q, n, k] <= vars["x_k"][n, k],
                        name=f"c2_{q}_{n}_{k}")


def add_c3(model, data, domains, vars):
    Q = data["Q"]
    for q in Q:
        model.addConstr(
            gp.quicksum(
                vars["a"][q, n, k]
                for (qq, n, k) in domains["a_domain"]
                if qq == q
            ) <= 1,
            name=f"c3_{q}",
        )


def add_c4(model, data, vars):
    K = data["K"]
    for k_idx, k in enumerate(K):
        inicio = data["V"][k_idx][0]
        fim = data["V"][k_idx][data["V_tamanho"][k_idx] - 1]
        model.addConstr(vars["x_k"][inicio, k] == 1,
                        name=f"c4_inicio_{k}")
        model.addConstr(vars["x_k"][fim, k] == 1,
                        name=f"c4_fim_{k}")


def add_c5(model, data, domains, vars):
    K = data["K"]
    V = data["V"]
    V_tamanho = data["V_tamanho"]
    D = data["D"]
    d_route_max = data["d_route_max"]

    for k_idx, k in enumerate(K):
        for idx in range(2, V_tamanho[k_idx] + 1):
            stop_node = V[k_idx][idx - 1]
            lhs = vars["x_k"][stop_node, k]

            rhs = (
                gp.quicksum(
                    vars["x_k"][V[k_idx][idx_g - 1], k]
                    for idx_g in range(1, idx)
                    if D[k_idx]                    # D[k_idx][n1-1][n2-1]
                    [V[k_idx][idx_g - 1] - 1]
                    [stop_node - 1]
                    <= d_route_max
                )
                + vars["s_k"][k, idx]
            )

            model.addConstr(lhs <= rhs, name=f"c5_{k}_{idx}")


def add_c6(model, data, domains, vars):
    K = data["K"]
    de = data["de"]

    for k_idx, k in enumerate(K):
        model.addConstr(
            gp.quicksum(
                vars["a"][q, n, k] * de[q - 1]
                for (q, n, kk) in domains["a_domain"]
                if kk == k
            ) <= vars["Cap"][k],
            name=f"c6_{k}",
        )


def add_c7(model, data, vars):
    N = data["N"]
    m_max = data["m_max"]

    for n in N:
        model.addConstr(
            gp.quicksum(
                vars["x_k"][n, k]
                for (nn, k) in data["I"]
                if nn == n
            ) <= m_max * vars["x"][n],
            name=f"c7_{n}",
        )


def add_c8(model, data, vars):
    K = data["K"]
    model.addConstr(
        gp.quicksum(vars["Cap"][k] for k in K)
        <= data["Capt"] + vars["Cad"],
        name="c8",
    )