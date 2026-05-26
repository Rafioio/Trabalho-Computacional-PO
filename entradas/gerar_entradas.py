import numpy as np
import random
import json
import os
from datetime import datetime

# ============================================
# CONFIGURAÇÕES GLOBAIS
# ============================================
NUM_N = 50              # Número de nós candidatos a parada (N)
NUM_Q = 35              # Número de nós de demanda (Q)

# ============================================
# PARÂMETROS REALISTAS
# ============================================
D_WALK_MAX = 500.0          # Distância máxima de caminhada (metros)
D_ROUTE_MAX = 800.0         # Distância máxima entre paradas consecutivas (metros)
P = 1000.0                  # Penalidade por não atendimento
CAPT = 800                  # Capacidade total disponível
M_MAX = 3                   # Número máximo de rotas por ponto
OMEGA = 50.0                # Razão custo instalação / custo frota

# Pesos da função objetivo (devem somar 1)
W1 = 0.35                   # Peso do custo social (f1)
W2 = 0.15                   # Peso da viabilidade técnica (f2)
W3 = 0.30                   # Peso do custo de infraestrutura (f3)
W4 = 0.20                   # Peso da penalidade de espaçamento (f4)

# Configurações de grid (simula uma cidade)
GRID_WIDTH = 3000           # metros
GRID_HEIGHT = 3000          # metros
DISTANCIA_MINIMA_PONTOS = 80  # metros (evita pontos muito próximos)

# ============================================
# FUNÇÕES AUXILIARES
# ============================================

def calcular_distancia_euclidiana(p1, p2):
    """Calcula distância euclidiana entre dois pontos."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def gerar_posicoes_realistas(n, largura, altura, min_dist=100):
    """Gera posições realistas para pontos. Garante distância mínima."""
    posicoes = []
    max_tentativas = 2000
    
    for _ in range(n):
        tentativa = 0
        while tentativa < max_tentativas:
            # Distribuição com clusterização suave
            if random.random() < 0.6 and len(posicoes) > 0:
                base = random.choice(posicoes)
                x = base[0] + random.gauss(0, largura * 0.08)
                y = base[1] + random.gauss(0, altura * 0.08)
            else:
                x = random.uniform(0, largura)
                y = random.uniform(0, altura)
            
            x = max(0, min(largura, x))
            y = max(0, min(altura, y))
            
            # Verificar distância mínima
            valido = True
            for px, py in posicoes:
                if calcular_distancia_euclidiana((x, y), (px, py)) < min_dist:
                    valido = False
                    break
            
            if valido:
                posicoes.append((x, y))
                break
            tentativa += 1
        
        if tentativa >= max_tentativas:
            posicoes.append((random.uniform(0, largura), random.uniform(0, altura)))
    
    return np.array(posicoes)

def gerar_demandas_realistas(posicoes_pontos, num_demandas, largura, altura, d_walk_max):
    """Gera nós de demanda próximos aos pontos."""
    posicoes_demandas = []
    niveis_demanda = []
    
    for _ in range(num_demandas):
        if random.random() < 0.8 and len(posicoes_pontos) > 0:
            ponto_ref = random.randint(0, len(posicoes_pontos) - 1)
            angulo = random.uniform(0, 2 * np.pi)
            raio = random.uniform(0, d_walk_max * 0.7)
            x = posicoes_pontos[ponto_ref][0] + raio * np.cos(angulo)
            y = posicoes_pontos[ponto_ref][1] + raio * np.sin(angulo)
        else:
            x = random.uniform(0, largura)
            y = random.uniform(0, altura)
        
        x = max(0, min(largura, x))
        y = max(0, min(altura, y))
        posicoes_demandas.append((x, y))
        
        # Demanda com variação realista
        demanda = np.random.lognormal(mean=4.0, sigma=0.8)
        demanda = max(5, min(200, demanda))
        niveis_demanda.append(int(demanda))
    
    return np.array(posicoes_demandas), np.array(niveis_demanda)

def calcular_qualidade_pontos(num_pontos):
    """Calcula a qualidade técnica de cada ponto (wn)."""
    qualidades = np.random.beta(a=2, b=2, size=num_pontos)
    qualidades = 0.2 + 0.8 * qualidades
    return qualidades

def calcular_matriz_distancias(posicoes_demandas, posicoes_pontos):
    """Calcula a matriz de distâncias d[q][n]."""
    num_q = len(posicoes_demandas)
    num_n = len(posicoes_pontos)
    d = np.zeros((num_q, num_n))
    
    for q in range(num_q):
        for n in range(num_n):
            d[q, n] = calcular_distancia_euclidiana(posicoes_demandas[q], posicoes_pontos[n])
    
    return d

# ============================================
# GERAR DADOS COMPLETOS
# ============================================

def gerar_dados_completos(fixar_semente=None, prefixo=""):
    """
    Gera todos os dados necessários para o arquivo JSON.
    
    Parâmetros:
    - fixar_semente: Se None, gera dados aleatórios diferentes a cada execução.
                     Se for um número, fixa a semente para reprodutibilidade.
    - prefixo: Prefixo para os nomes dos arquivos (ex: "teste1_", "cenario2_")
    """
    
    # Determinar semente
    if fixar_semente is not None:
        semente = fixar_semente
        print(f"\n🔒 Usando semente fixa: {semente} (dados reprodutíveis)")
    else:
        # Usar timestamp para gerar semente aleatória
        semente = int(datetime.now().timestamp() * 1000000) % (2**32)
        print(f"\n🎲 Usando semente aleatória: {semente} (dados diferentes a cada execução)")
    
    # Fixar sementes
    random.seed(semente)
    np.random.seed(semente)
    
    print("="*60)
    print("GERADOR DE DADOS PARA MODELO DE PARADAS DE ÔNIBUS")
    print("="*60)
    print(f"\nConfigurações:")
    print(f"  - Pontos candidatos: {NUM_N}")
    print(f"  - Nós de demanda: {NUM_Q}")
    print(f"  - Grid: {GRID_WIDTH}x{GRID_HEIGHT} m")
    print(f"  - Semente utilizada: {semente}")
    
    # 1. Gerar posições
    print("\n[1/5] Gerando posições dos pontos candidatos...")
    posicoes_pontos = gerar_posicoes_realistas(NUM_N, GRID_WIDTH, GRID_HEIGHT, DISTANCIA_MINIMA_PONTOS)
    
    # 2. Gerar demandas
    print("[2/5] Gerando nós de demanda...")
    posicoes_demandas, niveis_demanda = gerar_demandas_realistas(
        posicoes_pontos, NUM_Q, GRID_WIDTH, GRID_HEIGHT, D_WALK_MAX
    )
    
    # 3. Calcular qualidades
    print("[3/5] Calculando qualidade técnica dos pontos...")
    qualidades_pontos = calcular_qualidade_pontos(NUM_N)
    
    # 4. Calcular matriz de distâncias
    print("[4/5] Calculando matriz de distâncias...")
    d = calcular_matriz_distancias(posicoes_demandas, posicoes_pontos)
    
    # 5. Estatísticas
    print("[5/5] Calculando estatísticas...")
    distancias_validas = d[d <= D_WALK_MAX]
    perc_acessiveis = (len(distancias_validas) / d.size) * 100 if d.size > 0 else 0
    
    # Estrutura completa dos dados
    dados = {
        # Metadados
        'metadata': {
            'versao': '1.0',
            'data_geracao': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'semente': semente,
            'aleatorio': fixar_semente is None,
            'descricao': 'Dados de entrada para o problema de localizacao de paradas de onibus'
        },
        
        # Parâmetros do modelo
        'parametros': {
            'NumN': NUM_N,
            'NumQ': NUM_Q,
            'd_walk_max': D_WALK_MAX,
            'd_route_max': D_ROUTE_MAX,
            'P': P,
            'Capt': CAPT,
            'm_max': M_MAX,
            'omega': OMEGA,
            'W1': W1,
            'W2': W2,
            'W3': W3,
            'W4': W4
        },
        
        # Conjunto C (pontos ativos) - vazio na entrada
        'C': [],
        
        # Dados primários (como listas Python nativas)
        'de': niveis_demanda.tolist(),
        'w': qualidades_pontos.tolist(),
        
        # Matriz de distâncias (q x n)
        'd': d.tolist(),
        
        # Dados para visualização
        'visualizacao': {
            'posicoes_pontos': [[float(x), float(y)] for x, y in posicoes_pontos],
            'posicoes_demandas': [[float(x), float(y)] for x, y in posicoes_demandas],
            'qualidades_pontos': qualidades_pontos.tolist(),
            'niveis_demanda': niveis_demanda.tolist()
        },
        
        # Estatísticas
        'estatisticas': {
            'perc_acessiveis': float(perc_acessiveis),
            'dist_media_acessiveis': float(np.mean(distancias_validas)) if len(distancias_validas) > 0 else 0,
            'dist_minima': float(np.min(d)),
            'dist_maxima': float(np.max(d)),
            'demanda_total': int(np.sum(niveis_demanda)),
            'demanda_media': float(np.mean(niveis_demanda)),
            'qualidade_media': float(np.mean(qualidades_pontos)),
            'qualidade_min': float(np.min(qualidades_pontos)),
            'qualidade_max': float(np.max(qualidades_pontos))
        }
    }
    
    print(f"\n📊 Estatísticas geradas:")
    print(f"  - Pontos acessíveis: {perc_acessiveis:.1f}%")
    print(f"  - Demanda total: {dados['estatisticas']['demanda_total']} passageiros")
    print(f"  - Demanda média: {dados['estatisticas']['demanda_media']:.1f}")
    print(f"  - Qualidade média dos pontos: {dados['estatisticas']['qualidade_media']:.3f}")
    
    return dados, semente

# ============================================
# EXPORTAR PARA JSON
# ============================================

def exportar_para_json(dados, nome_arquivo='entrada_modelo.json'):
    """Exporta todos os dados para um único arquivo JSON."""
    
    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        json.dump(dados, f, indent=2, ensure_ascii=False)
    
    tamanho = os.path.getsize(nome_arquivo) / 1024  # KB
    print(f"\n💾 Arquivo '{nome_arquivo}' criado com sucesso! ({tamanho:.1f} KB)")

def exportar_para_csv(dados, nome_arquivo='coordenadas.csv'):
    """Exporta coordenadas para CSV (opcional, para compatibilidade)."""
    
    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        f.write("tipo,id,x,y,demanda,qualidade\n")
        
        # Pontos candidatos
        for i, (x, y) in enumerate(dados['visualizacao']['posicoes_pontos']):
            f.write(f"ponto,{i+1},{x:.2f},{y:.2f},,{dados['w'][i]:.4f}\n")
        
        # Nós de demanda
        for i, (x, y) in enumerate(dados['visualizacao']['posicoes_demandas']):
            f.write(f"demanda,{i+1},{x:.2f},{y:.2f},{dados['de'][i]},\n")
    
    print(f"💾 Arquivo '{nome_arquivo}' criado com sucesso!")

def salvar_semente(semente, nome_arquivo='ultima_semente.txt'):
    """Salva a semente usada para referência futura."""
    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        f.write(f"Semente utilizada: {semente}\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Para reproduzir estes dados, use: fixar_semente={semente}\n")
    print(f"💾 Semente salva em '{nome_arquivo}'")

# ============================================
# GERAR MÚLTIPLOS CENÁRIOS
# ============================================

def gerar_multiplos_cenarios(num_cenarios=5, prefixo_base="cenario"):
    """Gera múltiplos cenários diferentes para teste."""
    
    print("\n" + "="*60)
    print(f"GERANDO {num_cenarios} CENÁRIOS DIFERENTES")
    print("="*60)
    
    cenarios = []
    for i in range(1, num_cenarios + 1):
        print(f"\n--- Cenário {i} ---")
        dados, semente = gerar_dados_completos(fixar_semente=None)  # Sempre aleatório
        nome_arquivo = f"{prefixo_base}_{i:02d}.json"
        exportar_para_json(dados, nome_arquivo)
        cenarios.append({
            'cenario': i,
            'arquivo': nome_arquivo,
            'semente': semente,
            'demanda_total': dados['estatisticas']['demanda_total']
        })
    
    # Salvar resumo
    with open('resumo_cenarios.json', 'w', encoding='utf-8') as f:
        json.dump(cenarios, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("RESUMO DOS CENÁRIOS GERADOS")
    print("="*60)
    for c in cenarios:
        print(f"  {c['arquivo']}: semente={c['semente']}, demanda_total={c['demanda_total']}")
    
    return cenarios

# ============================================
# FUNÇÃO PRINCIPAL
# ============================================

def main():
    """Função principal com opções interativas."""
    
    print("="*60)
    print("GERADOR DE DADOS PARA MODELO DE PARADAS DE ÔNIBUS")
    print("="*60)
    
    print("\nOpções:")
    print("  1. Gerar um cenário aleatório (diferente a cada execução)")
    print("  2. Gerar um cenário com semente fixa (reprodutível)")
    print("  3. Gerar múltiplos cenários para teste")
    print("  4. Usar última semente salva (reproduzir último cenário)")
    
    opcao = input("\nEscolha uma opção (1-4): ").strip()
    
    if opcao == '1':
        # Cenário aleatório
        dados, semente = gerar_dados_completos(fixar_semente=None)
        exportar_para_json(dados, 'entrada_modelo.json')
        exportar_para_csv(dados, 'coordenadas.csv')
        salvar_semente(semente)
        
    elif opcao == '2':
        # Cenário com semente fixa
        try:
            semente = int(input("Digite a semente desejada (número inteiro): "))
        except:
            semente = 42
            print(f"Usando semente padrão: {semente}")
        dados, _ = gerar_dados_completos(fixar_semente=semente)
        exportar_para_json(dados, 'entrada_modelo.json')
        exportar_para_csv(dados, 'coordenadas.csv')
        salvar_semente(semente)
        
    elif opcao == '3':
        # Múltiplos cenários
        try:
            num = int(input("Quantos cenários deseja gerar? (padrão 5): ") or 5)
        except:
            num = 5
        gerar_multiplos_cenarios(num_cenarios=num)
        
    elif opcao == '4':
        # Reproduzir último cenário
        try:
            with open('ultima_semente.txt', 'r') as f:
                conteudo = f.read()
                import re
                match = re.search(r'Semente utilizada: (\d+)', conteudo)
                if match:
                    semente = int(match.group(1))
                    print(f"Reproduzindo cenário com semente: {semente}")
                    dados, _ = gerar_dados_completos(fixar_semente=semente)
                    exportar_para_json(dados, 'entrada_modelo.json')
                    exportar_para_csv(dados, 'coordenadas.csv')
                else:
                    print("Não foi possível ler a semente. Gerando cenário aleatório...")
                    dados, semente = gerar_dados_completos(fixar_semente=None)
                    exportar_para_json(dados, 'entrada_modelo.json')
                    exportar_para_csv(dados, 'coordenadas.csv')
        except FileNotFoundError:
            print("Nenhuma semente salva encontrada. Gerando cenário aleatório...")
            dados, semente = gerar_dados_completos(fixar_semente=None)
            exportar_para_json(dados, 'entrada_modelo.json')
            exportar_para_csv(dados, 'coordenadas.csv')
            salvar_semente(semente)
    
    else:
        print("Opção inválida! Gerando cenário aleatório...")
        dados, semente = gerar_dados_completos(fixar_semente=None)
        exportar_para_json(dados, 'entrada_modelo.json')
        exportar_para_csv(dados, 'coordenadas.csv')
        salvar_semente(semente)
    
    print("\n" + "="*60)
    print("✅ GERAÇÃO CONCLUÍDA!")
    print("="*60)
    print("\nArquivos gerados:")
    print("  📄 entrada_modelo.json - Dados completos (usar no otimizador)")
    print("  📄 coordenadas.csv - Coordenadas para Excel")
    print("  📄 ultima_semente.txt - Semente usada (para reprodução)")
    print("\n💡 Dica: Execute novamente para gerar um conjunto de dados diferente!")

if __name__ == "__main__":
    main()