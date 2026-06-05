import gurobipy as gp
from gurobipy import GRB


def build_objective_exprs(data, vars, domains):
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

    a_domain = domains["a_domain"]
    S_Indices = domains["S_Indices"]

    # f1: Custo social (Caminhada + Exclusão)
    f1 = gp.quicksum(
        de[q - 1] * (
            gp.quicksum(d[q - 1][n - 1] * vars["a"][q, n, k] for (qq, n, k) in a_domain if qq == q)
            + (1 - gp.quicksum(vars["a"][q, n, k] for (qq, n, k) in a_domain if qq == q)) * P
        )
        for q in Q
    )

    # f2: Penalidade de espaçamento (Corrigido para bater com Eq. 2 do PDF)
    f2 = gp.quicksum(vars["s_k"][k, idx] for (k, idx) in S_Indices)

    # f3: Custo de infraestrutura (Corrigido para bater com Eq. 3 do PDF)
    f3 = omega * vars["Cad"] + gp.quicksum(vars["x"][n] for n in T)

    # f4: Viabilidade técnica (Corrigido para incluir Cap[k], batendo com Eq. 4 do PDF)
    f4 = gp.quicksum((1 / w[n - 1]) * (vars["x_k"][n, k] * vars["Cap"][k]) for (n, k) in I)

    return {"f1": f1, "f2": f2, "f3": f3, "f4": f4}


def build_objective(model, data, vars, domains):
    exprs = build_objective_exprs(data, vars, domains)
    
    # 1. Recupera os hiperparâmetros (com valores fallback de 0.5 por segurança)
    W1 = data.get("W1", 0.5)
    W2 = data.get("W2", 0.5)
    W3 = data.get("W3", 0.5)
    W4 = data.get("W4", 0.5)
    
    mu = data.get("mu", 0.5)
    theta = data.get("theta", 0.5)
    
    # 2. Recupera os limites de normalização calculados pelo weight_normalizer.py
    f1_min, f1_max = data.get("f1_min", 0.0), data.get("f1_max", 1.0)
    f2_min, f2_max = data.get("f2_min", 0.0), data.get("f2_max", 1.0)
    f3_min, f3_max = data.get("f3_min", 0.0), data.get("f3_max", 1.0)
    f4_min, f4_max = data.get("f4_min", 0.0), data.get("f4_max", 1.0)

    # Função local para normalizar os valores brutos entre 0 e 1
    def normalizar(f_expr, f_min, f_max):
        denominador = f_max - f_min
        if denominador < 1e-6: # Proteção contra divisão por zero (se min == max)
            return f_expr - f_min
        return (f_expr - f_min) / denominador

    # 3. Aplica a fórmula Min-Max em todas as funções (f - f_min) / (f_max - f_min)
    f1_norm = normalizar(exprs["f1"], f1_min, f1_max)
    f2_norm = normalizar(exprs["f2"], f2_min, f2_max)
    f3_norm = normalizar(exprs["f3"], f3_min, f3_max)
    f4_norm = normalizar(exprs["f4"], f4_min, f4_max)

    # 4. Constrói os Macro-Objetivos do Projeto SouBus (Eqs. 5 e 6 do PDF)
    F_usuario = W1 * f1_norm + W2 * f2_norm
    F_operador = W3 * f3_norm + W4 * f4_norm

    # 5. Define a Função Objetivo Global balanceada (Eq. 7 do PDF)
    model.setObjective(mu * F_usuario + theta * F_operador, GRB.MINIMIZE)
    
    # Salva os macro-objetivos no dicionário caso queira imprimi-los depois
    exprs["F_usuario"] = F_usuario
    exprs["F_operador"] = F_operador
    
    return exprs