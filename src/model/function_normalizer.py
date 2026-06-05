from gurobipy import GRB

from src.model.solver import build_model


OBJ_NAMES = ["f1", "f2", "f3", "f4"]
OBJ_LABELS = {
    "f1": "custo social",
    "f2": "penalidade de espaçamento",
    "f3": "custo de infraestrutura",
    "f4": "viabilidade técnica",
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


def normalize_function(data, verbose=False):
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

    for j in range(4):
        vals = [payoff[i][j] for i in range(4) if payoff[i][j] is not None]
        if len(vals) < 2:
            if verbose:
                print(f"\n  AVISO: {OBJ_NAMES[j]} — dados insuficientes para limites.")
            # Valores padrão de segurança (fallback) caso o solver não ache solução viável na matriz
            utopia.append(0.0)
            anti_utopia.append(1.0)
            continue

        u = min(vals)
        a = max(vals)
        
        # Proteção matemática: se o min e o max forem iguais (o que significa 
        # que a função não variou na matriz), o denominador da fórmula seria zero.
        # Adicionamos uma pequena margem para evitar erro de divisão por zero no objective.py.
        if abs(a - u) < 1e-6:
            a = u + 1.0

        utopia.append(u)
        anti_utopia.append(a)

    if verbose:
        print(f"\n{'='*60}")
        print("Limites Calculados para Normalização Min-Max das Funções")
        print(f"{'='*60}")
        header = "".join(f"{name:>18}" for name in OBJ_NAMES)
        print(f"{'':>18}{header}")
        u_row = "".join(f"{utopia[j]:>18.4f}" for j in range(4))
        a_row = "".join(f"{anti_utopia[j]:>18.4f}" for j in range(4))
        print(f"{'min (utopia)':>18}{u_row}")
        print(f"{'max (nadir)':>18}{a_row}")

    # =====================================================================
    # INJEÇÃO DOS LIMITES NO DICIONÁRIO 'DATA' PARA USO NO OBJECTIVE.PY
    # =====================================================================
    data["f1_min"] = utopia[0]
    data["f1_max"] = anti_utopia[0]
    
    data["f2_min"] = utopia[1]
    data["f2_max"] = anti_utopia[1]
    
    data["f3_min"] = utopia[2]
    data["f3_max"] = anti_utopia[2]
    
    data["f4_min"] = utopia[3]
    data["f4_max"] = anti_utopia[3]

    data["_normalization"] = {
        "payoff": payoff,
        "utopia": utopia,
        "anti_utopia": anti_utopia,
    }