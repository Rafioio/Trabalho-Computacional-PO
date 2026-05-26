from gurobipy import GRB

from src.model.solver import build_model


OBJ_NAMES = ["f1", "f2", "f3", "f4"]
OBJ_LABELS = {
    "f1": "custo social",
    "f2": "viabilidade técnica",
    "f3": "custo de infraestrutura",
    "f4": "penalidade de espaçamento",
}


def compute_payoff_table(data, verbose=False):
    model, vars, domains, obj_exprs = build_model(data)

    obj_list = [obj_exprs[name] for name in OBJ_NAMES]
    payoff = [[None] * 4 for _ in range(4)]

    for i, name_i in enumerate(OBJ_NAMES):
        if verbose:
            print(f"\n  Resolvendo para {name_i} ({OBJ_LABELS[name_i]})...")

        model.setObjective(obj_list[i], GRB.MINIMIZE)
        model.Params.OutputFlag = 1 if verbose else 0
        model.optimize()

        if model.status != GRB.OPTIMAL:
            if verbose:
                print(f"  AVISO: {name_i} não convergiu (status {model.status})")
            for j in range(4):
                payoff[i][j] = None
            continue

        for j, name_j in enumerate(OBJ_NAMES):
            val = obj_exprs[name_j].getValue()
            payoff[i][j] = val
            if verbose:
                print(f"    {name_j} = {val:.4f}")

    return payoff


def normalize_weights(data, verbose=False):
    payoff = compute_payoff_table(data, verbose=verbose)

    if verbose:
        print(f"\n{'='*60}")
        print("Matriz payoff (linha = obj minimizado, coluna = valor obtido)")
        print(f"{'='*60}")
        header = "".join(f"{name:>18}" for name in OBJ_NAMES)
        print(f"{'':>18}{header}")
        for i, name_i in enumerate(OBJ_NAMES):
            row = "".join(f"{payoff[i][j]:>18.4f}" if payoff[i][j] is not None else f"{'---':>18}" for j in range(4))
            print(f"{name_i:>18}{row}")

    utopia = []
    anti_utopia = []
    factors = []
    original_weights = [data["W1"], data["W2"], data["W3"], data["W4"]]

    for j in range(4):
        vals = [payoff[i][j] for i in range(4) if payoff[i][j] is not None]
        if len(vals) < 2:
            if verbose:
                print(f"\n  AVISO: {OBJ_NAMES[j]} — dados insuficientes, fator = 1.0")
            utopia.append(None)
            anti_utopia.append(None)
            factors.append(1.0)
            continue

        u = min(vals)
        a = max(vals)
        r = a - u
        f = 1.0 / r if r > 1e-10 else 1.0

        utopia.append(u)
        anti_utopia.append(a)
        factors.append(f)

    if verbose:
        print(f"\n{'='*60}")
        print("Pontos utópico e anti-utópico")
        print(f"{'='*60}")
        header = "".join(f"{name:>18}" for name in OBJ_NAMES)
        print(f"{'':>18}{header}")
        u_row = "".join(f"{utopia[j]:>18.4f}" if utopia[j] is not None else f"{'---':>18}" for j in range(4))
        a_row = "".join(f"{anti_utopia[j]:>18.4f}" if anti_utopia[j] is not None else f"{'---':>18}" for j in range(4))
        f_row = "".join(f"{factors[j]:>18.6f}" for j in range(4))
        print(f"{'utopia':>18}{u_row}")
        print(f"{'anti-utopia':>18}{a_row}")
        print(f"{'fator':>18}{f_row}")

    new_weights = [original_weights[j] * factors[j] for j in range(4)]

    if verbose:
        print(f"\n{'='*60}")
        print("Normalização dos pesos")
        print(f"{'='*60}")
        ow_row = "".join(f"{original_weights[j]:>18.6f}" for j in range(4))
        nw_row = "".join(f"{new_weights[j]:>18.6f}" for j in range(4))
        print(f"{'original':>18}{ow_row}")
        print(f"{'normalizado':>18}{nw_row}")

    data["W1"] = new_weights[0]
    data["W2"] = new_weights[1]
    data["W3"] = new_weights[2]
    data["W4"] = new_weights[3]
    data["_normalization"] = {
        "payoff": payoff,
        "utopia": utopia,
        "anti_utopia": anti_utopia,
        "factors": factors,
    }