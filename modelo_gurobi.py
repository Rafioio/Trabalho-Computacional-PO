# Conversão do modelo `.mod` para `gurobipy`
import gurobipy as gp
from gurobipy import GRB

# ==========================================================
# Modelo
# ==========================================================

model = gp.Model("modelo_transporte")

# ==========================================================
# CONJUNTOS E DADOS
# ==========================================================
# Você precisa carregar os dados do .dat aqui.
# O ideal é transformar o .dat em JSON, CSV ou Pickle.
# Abaixo estão os formatos esperados.

Q = [...]                 # demandas
N = [...]                 # nós/pontos
K = [...]                 # rotas
T = [...]                 # subconjunto de N
I = [...]                 # pares (n,k)
L = [...]                 # pares (q,k)

# parâmetros

de = {}                   # de[q]
d = {}                    # d[q,n]
w = {}                    # w[n]
D = {}                    # D[k,n1,n2]
V = {}                    # V[k][idx]
V_tamanho = {}

P = 1000
omega = 1.0

W1 = 1e-3
W2 = 1.0
W3 = 0.94
W4 = 1.0

Capt = 8000
m_max = 7

d_route_max = 250.0
d_walk_max = 100.0

# ==========================================================
# DOMÍNIO DISPERSO
# ==========================================================

a_domain = []

for q in Q:
    for (n, k) in I:
        if (q, k) in L:
            if d[q, n] <= d_walk_max:
                a_domain.append((q, n, k))

S_Indices = []

for k in K:
    for idx in range(2, V_tamanho[k] + 1):
        S_Indices.append((k, idx))

# ==========================================================
# VARIÁVEIS
# ==========================================================

x = model.addVars(N, vtype=GRB.BINARY, name="x")

x_k = model.addVars(I, vtype=GRB.BINARY, name="x_k")

a = model.addVars(
    a_domain,
    lb=0,
    ub=1,
    vtype=GRB.CONTINUOUS,
    name="a"
)

Cap = model.addVars(K, vtype=GRB.INTEGER, lb=0, name="Cap")

Cad = model.addVar(vtype=GRB.INTEGER, lb=0, name="Cad")

s_k = model.addVars(
    S_Indices,
    vtype=GRB.CONTINUOUS,
    lb=0,
    name="s_k"
)

# ==========================================================
# FUNÇÃO OBJETIVO
# ==========================================================

f1 = gp.quicksum(
    de[q] * (
        gp.quicksum(
            d[q, n] * a[q, n, k]
            for (qq, n, k) in a_domain
            if qq == q
        )
        +
        (
            1
            - gp.quicksum(
                a[q, n, k]
                for (qq, n, k) in a_domain
                if qq == q
            )
        ) * P
    )
    for q in Q
)

f2 = gp.quicksum(
    (1 / w[n]) * x_k[n, k]
    for (n, k) in I
)

f3 = omega * Cad + gp.quicksum(x[n] for n in T)

f4 = gp.quicksum(s_k[k, idx] for (k, idx) in S_Indices)

model.setObjective(
    W1 * f1 +
    W2 * f2 +
    W3 * f3 +
    W4 * f4,
    GRB.MINIMIZE
)

# ==========================================================
# RESTRIÇÕES
# ==========================================================

# ----------------------------------------------------------
# (1) Ativação da infraestrutura
# ----------------------------------------------------------

for (n, k) in I:
    model.addConstr(
        x_k[n, k] <= x[n]
    )

# ----------------------------------------------------------
# (2) Viabilidade de embarque
# ----------------------------------------------------------

for (q, n, k) in a_domain:
    model.addConstr(
        a[q, n, k] <= x_k[n, k]
    )

# ----------------------------------------------------------
# (3) Conservação da demanda
# ----------------------------------------------------------

for q in Q:
    model.addConstr(
        gp.quicksum(
            a[q, n, k]
            for (qq, n, k) in a_domain
            if qq == q
        ) <= 1
    )

# ----------------------------------------------------------
# (4) Terminais obrigatórios
# ----------------------------------------------------------

for k in K:

    inicio = V[k][1]
    fim = V[k][V_tamanho[k]]

    model.addConstr(x_k[inicio, k] == 1)
    model.addConstr(x_k[fim, k] == 1)

# ----------------------------------------------------------
# (5) Espaçamento máximo com folga
# ----------------------------------------------------------

for (k, idx) in S_Indices:

    lhs = x_k[V[k][idx], k]

    rhs = gp.quicksum(
        x_k[V[k][idx_gamma], k]
        for idx_gamma in range(1, idx)
        if D[k, V[k][idx_gamma], V[k][idx]] <= d_route_max
    ) + s_k[k, idx]

    model.addConstr(lhs <= rhs)

# ----------------------------------------------------------
# (6) Capacidade da rota
# ----------------------------------------------------------

for k in K:

    model.addConstr(
        gp.quicksum(
            a[q, n, k] * de[q]
            for (q, n, kk) in a_domain
            if kk == k
        ) <= Cap[k]
    )

# ----------------------------------------------------------
# (7) Limite operacional do ponto
# ----------------------------------------------------------

for n in N:

    model.addConstr(
        gp.quicksum(
            x_k[n, k]
            for (nn, k) in I
            if nn == n
        ) <= m_max * x[n]
    )

# ----------------------------------------------------------
# (8) Capacidade total do sistema
# ----------------------------------------------------------

model.addConstr(
    gp.quicksum(Cap[k] for k in K)
    <= Capt + Cad
)

# ==========================================================
# OTIMIZAÇÃO
# ==========================================================

model.optimize()

# ==========================================================
# RESULTADOS
# ==========================================================

if model.status == GRB.OPTIMAL:

    print("Valor da função objetivo:", model.objVal)

    print("f1:", f1.getValue())
    print("f2:", f2.getValue())
    print("f3:", f3.getValue())
    print("f4:", f4.getValue())

    print("\nCapacidade das rotas:")

    for k in K:
        print(f"Rota {k}: {Cap[k].X}")

    print("\nCapacidade adicional:", Cad.X)

else:
    print("Modelo não encontrou solução ótima.")

