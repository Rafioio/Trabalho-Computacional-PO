import math
import random
import gurobipy as gp

# Trava a semente para que a instância de 50 pontos seja sempre idêntica em todos os testes
random.seed(42)

# =================================================================
# 1. DIMENSÕES DA INSTÂNCIA (MOCK COERENTE)
# =================================================================
NumN = 50   # 50 pontos candidatos na cidade
NumK = 5    # 5 rotas de ônibus
NumQ = 10   # 10 zonas de demanda populacional

# Parâmetros Escalares do PDF
omega = 1.0          
P = 1000.0           
Capt = 1200         
m_max = 7            
d_route_max = 600.0  # Limite de distância entre paradas
d_walk_max = 400.0   # Limite de caminhada
INF_DIST = float('inf')

Q = list(range(1, NumQ + 1))
N = list(range(1, NumN + 1))
K = list(range(1, NumK + 1))

# =================================================================
# 2. GERAÇÃO ESPACIAL COERENTE (PLANO CARTESIANO 2km x 2km)
# =================================================================
# Gera coordenadas (X, Y) em metros para os nós e demandas
coords_N = {n: (random.uniform(0, 2000), random.uniform(0, 2000)) for n in N}
coords_Q = {q: (random.uniform(200, 1800), random.uniform(200, 1800)) for q in Q}

# C: Pontos que já possuem abrigo de ônibus (aprox. 20% da cidade)
C = {n for n in N if random.random() < 0.20}
T = [n for n in N if n not in C]

# Vetores de Demanda (de) e Viabilidade (w)
de = {q: float(random.randint(50, 300)) for q in Q}
w = {n: round(random.uniform(0.5, 1.0), 2) for n in N}

# =================================================================
# 3. CONSTRUÇÃO DA MATRIZ DE CAMINHADA (d)
# =================================================================
d = {}
for q in Q:
    xq, yq = coords_Q[q]
    for n in N:
        xn, yn = coords_N[n]
        dist = math.sqrt((xq - xn)**2 + (yq - yn)**2)
        # Mantém esparso: só salva distâncias que não sejam absurdas
        if dist <= d_walk_max * 1.5:  
            d[(q, n)] = round(dist, 2)

# =================================================================
# 4. CONSTRUÇÃO DE ROTAS (V) E MATRIZ VIÁRIA (D)
# =================================================================
V = {}
V_tamanho = {}
D = {}
I_list = []

for k in K:
    # Cada rota terá entre 8 e 15 pontos candidatos
    tamanho = random.randint(8, 15)
    V_tamanho[k] = tamanho
    
    # Sorteia os nós e ordena pelo eixo X para simular uma linha reta/lógica na cidade
    nos_rota = random.sample(N, tamanho)
    nos_rota.sort(key=lambda n: coords_N[n][0])
    
    for pos, no in enumerate(nos_rota, start=1):
        V[(k, pos)] = no
        I_list.append((no, k))
        
    # Calcula a distância viária cumulativa D[k, i, j] ao longo da rota
    for i in range(tamanho):
        no_origem = nos_rota[i]
        distancia_acumulada = 0.0
        for j in range(i + 1, tamanho):
            no_destino = nos_rota[j]
            no_anterior = nos_rota[j - 1]
            
            # Distância do trecho anterior até o atual
            x1, y1 = coords_N[no_anterior]
            x2, y2 = coords_N[no_destino]
            distancia_acumulada += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            D[(k, no_origem, no_destino)] = round(distancia_acumulada, 2)

I = gp.tuplelist(I_list)

# =================================================================
# 5. CONSTRUÇÃO DE LIGACÕES DEMANDA-ROTA (L)
# =================================================================
L_list = []
for q in Q:
    for k in K:
        # Se a rota k passa perto da demanda q (dentro do walk_max), eles se conectam
        atende = any(
            d.get((q, V[(k, pos)]), 99999) <= d_walk_max 
            for pos in range(1, V_tamanho[k] + 1)
        )
        if atende:
            L_list.append((q, k))

L = gp.tuplelist(L_list)
L_set = set(L)

print("Módulo dados.py carregado: Instância espacial coerente de 50 nós gerada com sucesso!")