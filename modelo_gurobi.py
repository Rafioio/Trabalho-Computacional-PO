import gurobipy as gp
from gurobipy import GRB
from dados import (
    N, Q, K, T, d, de, w, d_walk_max, d_route_max, V, V_tamanho, D, I, L_set, Capt, m_max, P, omega, INF_DIST
)

# Valores Utopia (min) e Nadir (max) obtidos via otimizações preliminares.
# Preenchidos com 0.0 e 1.0 temporariamente para manter o modelo executável e evitar divisão por zero.

W1, W2 = 0.5, 0.5    
mu = 0.1             
theta = 0.9          

f1_min, f1_max = 0.0, 1.0
f2_min, f2_max = 0.0, 1.0
f3_min, f3_max = 0.0, 1.0
f4_min, f4_max = 0.0, 1.0

# =================================================================
# 3. DOMÍNIOS DISPERSOS (Preservando a Lógica OPL)
# =================================================================

a_domain = []
a_por_demanda = {q: [] for q in Q}
a_por_rota = {k: [] for k in K}

for (n, k) in I:
    for q in Q:
        if (q, k) in L_set and d.get((q, n), INF_DIST) <= d_walk_max:
            a_domain.append((q, n, k))
            a_por_demanda[q].append((n, k))
            a_por_rota[k].append((q, n))

S_Indices = []
for k in K:
    if k in V_tamanho and V_tamanho[k] >= 2:
        for idx_zero_based in range(1, V_tamanho[k]):
            S_Indices.append((k, idx_zero_based))

# =================================================================
# 4. VARIÁVEIS DE DECISÃO
# =================================================================

model = gp.Model("Localizacao_Paradas_Onibus")

# Habilitar MIQP/MIQCQP para aceitar a multiplicação Quadrática na f4
model.Params.NonConvex = 2

x = model.addVars(N, vtype=GRB.BINARY, name="x")
x_k = model.addVars(I, vtype=GRB.BINARY, name="x_k")
a_var = model.addVars(a_domain, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="a")
Cap = model.addVars(K, vtype=GRB.INTEGER, lb=0, name="Cap")
Cad = model.addVar(vtype=GRB.INTEGER, lb=0, name="Cad")
s_k = model.addVars(S_Indices, vtype=GRB.CONTINUOUS, lb=0.0, name="s_k")

# =================================================================
# 5. FUNÇÕES OBJETIVO
# =================================================================

# f1: Custo social de caminhada e exclusão (Eq. 7 do PDF)
f1 = gp.quicksum(
    de[q] * (
        gp.quicksum(d[q, n] * a_var[q, n, k] for (n, k) in a_por_demanda[q])
        + 
        (1.0 - gp.quicksum(a_var[q, n, k] for (n, k) in a_por_demanda[q])) * P
    )
    for q in Q
)

# f2: Quebras de espaçamento (Eq. 8 do PDF)
f2 = gp.quicksum(s_k[k, idx] for (k, idx) in S_Indices)

# f3: Custo de infraestrutura e expansão (Eq. 9 do PDF)
f3 = (omega * Cad) + gp.quicksum(x[n] for n in T)

# f4: Viabilidade técnica - Modelo MIQP: Binária (x_k) * Inteira (Cap) (Eq. 10 do PDF)
f4 = gp.quicksum((1.0 / w[n]) * (x_k[n, k] * Cap[k]) for (n, k) in I)

# =================================================================
# MACRO-OBJETIVOS E NORMALIZAÇÃO (Eq. 11, 12 e 13 do PDF)
# =================================================================

F_usuario = mu * ((f1 - f1_min) / (f1_max - f1_min)) + (1.0 - mu) * ((f2 - f2_min) / (f2_max - f2_min))
F_operador = theta * ((f3 - f3_min) / (f3_max - f3_min)) + (1.0 - theta) * ((f4 - f4_min) / (f4_max - f4_min))

# Minimização da Função Objetivo Global unificada
model.setObjective(W1 * F_usuario + W2 * F_operador, GRB.MINIMIZE)

# =================================================================
# 6. RESTRIÇÕES DO MODELO
# =================================================================

# R1: Ativação da infraestrutura
for (n, k) in I:
    model.addConstr(x_k[n, k] <= x[n], name=f"R1_AtivInfra_{n}_{k}")

# R2: Viabilidade de embarque
for (q, n, k) in a_domain:
    model.addConstr(a_var[q, n, k] <= x_k[n, k], name=f"R2_Embarque_{q}_{n}_{k}")

# R3: Conservação da demanda
for q in Q:
    model.addConstr(
        gp.quicksum(a_var[q, n, k] for (n, k) in a_por_demanda[q]) <= 1.0, 
        name=f"R3_ConsDemanda_{q}"
    )

# R4: Terminais obrigatórios
for k in K:
    if k in V_tamanho and V_tamanho[k] >= 1:
        no_inicio = V.get((k, 1))
        no_fim = V.get((k, V_tamanho[k]))
        
        if no_inicio is not None and (no_inicio, k) in I:
            model.addConstr(x_k[no_inicio, k] == 1, name=f"R4_TermInicio_{k}")
        if no_fim is not None and (no_fim, k) in I:
            model.addConstr(x_k[no_fim, k] == 1, name=f"R4_TermFim_{k}")

# R5: Espaçamento máximo com folga
for (k, idx_zero) in S_Indices:
    pos_atual_opl = idx_zero + 1  
    atual_no = V.get((k, pos_atual_opl))
    
    if atual_no is not None and (atual_no, k) in I:
        candidatos = []
        for pos_anterior_opl in range(1, pos_atual_opl):
            anterior_no = V.get((k, pos_anterior_opl))
            if anterior_no is not None and (anterior_no, k) in I:
                if D.get((k, anterior_no, atual_no), INF_DIST) <= d_route_max:
                    candidatos.append(x_k[anterior_no, k])
        
        model.addConstr(
            x_k[atual_no, k] <= gp.quicksum(candidatos) + s_k[k, idx_zero], 
            name=f"R5_Espacamento_{k}_{pos_atual_opl}"
        )

# R6: Capacidade da rota
for k in K:
    model.addConstr(
        gp.quicksum(a_var[q, n, k] * de[q] for (q, n) in a_por_rota[k]) <= Cap[k], 
        name=f"R6_CapRota_{k}"
    )

# R7: Limite operacional do ponto
for n in N:
    model.addConstr(
        x_k.sum(n, '*') <= m_max * x[n], 
        name=f"R7_LimitePonto_{n}"
    )

# R8: Capacidade total do sistema
model.addConstr(
    gp.quicksum(Cap[k] for k in K) <= Capt + Cad, 
    name="R8_CapTotalSistema"
)

# =================================================================
# 7. EXECUÇÃO E EXIBIÇÃO DOS RESULTADOS
# =================================================================

model.optimize()

if model.status == GRB.OPTIMAL:
    print("\n" + "="*50)
    print("OTIMIZAÇÃO CONCLUÍDA COM SUCESSO (Solução Ótima Encontrada)")
    print("="*50)
    print(f"Função Objetivo Global (F): {model.ObjVal:.4f}")
    print(f"F_usuario: {F_usuario.getValue():.4f}")
    print(f"F_operador: {F_operador.getValue():.4f}")
    print("-" * 50)
    print(f"f1 (Custo Social): {f1.getValue():.4f}")
    print(f"f2 (Penalidade de Espaçamento): {f2.getValue():.4f}")
    print(f"f3 (Custo de Infraestrutura): {f3.getValue():.4f}")
    print(f"f4 (Viabilidade Técnica): {f4.getValue():.4f}")
    
    print("\nDimensionamento da Capacidade por Rota:")
    for k in K:
        print(f" - Rota {k}: {Cap[k].X:.0f} ônibus alocados")
        
    print(f"\nAlocação Extraordinária (Cad): {Cad.X:.0f} ônibus")
else:
    print(f"\nO modelo encerrou a otimização sem atingir a otimalidade. Código de Status Gurobi: {model.status}")