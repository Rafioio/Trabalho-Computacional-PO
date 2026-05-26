import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import os

# ============================================
# FUNÇÕES PARA LER OS DADOS
# ============================================

def ler_dados_json(nome_arquivo='cenario_01.json'):
    """Lê todos os dados do arquivo JSON."""
    try:
        with open(nome_arquivo, 'r', encoding='utf-8') as f:
            dados = json.load(f)
        return dados
    except FileNotFoundError:
        print(f"Erro: Arquivo '{nome_arquivo}' não encontrado!")
        print("Execute primeiro o gerador de dados.")
        return None

def ler_solucao_json(nome_arquivo='solucao.json'):
    """Lê a solução do modelo (se existir)."""
    try:
        with open(nome_arquivo, 'r', encoding='utf-8') as f:
            solucao = json.load(f)
        return solucao
    except FileNotFoundError:
        return None

# ============================================
# FUNÇÕES DE VISUALIZAÇÃO
# ============================================

def calcular_limites_adaptativos(posicoes, padding_percent=0.12):
    """Calcula limites adaptativos para o gráfico baseado nas posições."""
    if len(posicoes) == 0:
        return (0, 10, 0, 10)
    
    x_coords = [p[0] for p in posicoes]
    y_coords = [p[1] for p in posicoes]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Adicionar padding percentual
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    if x_range == 0:
        x_range = 1
    if y_range == 0:
        y_range = 1
    
    x_padding = x_range * padding_percent
    y_padding = y_range * padding_percent
    
    return (x_min - x_padding, x_max + x_padding, 
            y_min - y_padding, y_max + y_padding)

def desenhar_grid_adaptativo(ax, x_min, x_max, y_min, y_max):
    """Desenha um grid adaptativo com escala automática."""
    
    # Determinar número de divisões baseado na escala
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    num_divisoes_x = max(5, min(10, int(x_range / 300)))
    num_divisoes_y = max(5, min(10, int(y_range / 300)))
    
    # Criar ticks para o grid
    x_ticks = np.linspace(x_min, x_max, num_divisoes_x + 1)
    y_ticks = np.linspace(y_min, y_max, num_divisoes_y + 1)
    
    # Desenhar linhas do grid
    for x in x_ticks:
        ax.axvline(x=x, color='lightgray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
    for y in y_ticks:
        ax.axhline(y=y, color='lightgray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
    
    # Configurar ticks
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis='both', labelsize=8)

def desenhar_barra_escala(ax, x_min, x_max, y_min, y_max):
    """Desenha uma barra de escala no canto inferior direito."""
    
    largura_plot = x_max - x_min
    altura_plot = y_max - y_min
    
    # Calcular tamanho da escala (15% da largura do gráfico)
    tamanho_escala = largura_plot * 0.15
    
    # Arredondar para um número bonito
    if tamanho_escala > 1000:
        tamanho_escala = round(tamanho_escala / 500) * 500
    elif tamanho_escala > 500:
        tamanho_escala = round(tamanho_escala / 100) * 100
    elif tamanho_escala > 200:
        tamanho_escala = round(tamanho_escala / 50) * 50
    elif tamanho_escala > 100:
        tamanho_escala = round(tamanho_escala / 25) * 25
    elif tamanho_escala > 50:
        tamanho_escala = round(tamanho_escala / 10) * 10
    else:
        tamanho_escala = round(tamanho_escala / 5) * 5
    
    if tamanho_escala < 10:
        tamanho_escala = round(tamanho_escala, 1)
    
    # Posição da escala
    x_start = x_max - largura_plot * 0.12
    x_end = x_start + tamanho_escala
    y_scale = y_min + altura_plot * 0.05
    
    # Desenhar linha principal
    ax.plot([x_start, x_end], [y_scale, y_scale], 'k-', linewidth=3, zorder=10)
    
    # Desenhar extremidades
    ax.plot([x_start, x_start], [y_scale - altura_plot*0.008, y_scale + altura_plot*0.008], 'k-', linewidth=2, zorder=10)
    ax.plot([x_end, x_end], [y_scale - altura_plot*0.008, y_scale + altura_plot*0.008], 'k-', linewidth=2, zorder=10)
    
    # Adicionar texto
    ax.text(x_start + tamanho_escala/2, y_scale - altura_plot*0.015, 
            f'{tamanho_escala} m', ha='center', va='top', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='gray'))

def desenhar_rosa_dos_ventos(ax, x_min, x_max, y_min, y_max):
    """Desenha uma rosa dos ventos no canto superior direito."""
    
    largura_plot = x_max - x_min
    altura_plot = y_max - y_min
    
    centro_x = x_max - largura_plot * 0.08
    centro_y = y_max - altura_plot * 0.08
    tamanho = min(largura_plot, altura_plot) * 0.04
    
    # Desenhar círculo
    circulo = Circle((centro_x, centro_y), tamanho, fill=False, edgecolor='#333333', linewidth=1.5, zorder=10)
    ax.add_patch(circulo)
    
    # Desenhar linhas
    ax.plot([centro_x, centro_x], [centro_y - tamanho, centro_y + tamanho], '#333333', linewidth=1.2, zorder=10)
    ax.plot([centro_x - tamanho, centro_x + tamanho], [centro_y, centro_y], '#333333', linewidth=1.2, zorder=10)
    
    # Adicionar textos
    ax.annotate('N', xy=(centro_x, centro_y + tamanho), xytext=(0, 3),
                textcoords='offset points', ha='center', va='bottom', 
                fontsize=9, fontweight='bold', color='#333333')
    ax.annotate('S', xy=(centro_x, centro_y - tamanho), xytext=(0, -3),
                textcoords='offset points', ha='center', va='top', 
                fontsize=8, color='#333333')
    ax.annotate('L', xy=(centro_x + tamanho, centro_y), xytext=(3, 0),
                textcoords='offset points', ha='left', va='center', 
                fontsize=8, color='#333333')
    ax.annotate('O', xy=(centro_x - tamanho, centro_y), xytext=(-3, 0),
                textcoords='offset points', ha='right', va='center', 
                fontsize=8, color='#333333')

def visualizar_pontos(dados_json, solucao=None):
    """Cria visualização adaptativa com todos os pontos."""
    
    if dados_json is None:
        return None, None
    
    # Extrair dados
    params = dados_json['parametros']
    visual = dados_json['visualizacao']
    estatisticas = dados_json['estatisticas']
    
    posicoes_pontos = np.array(visual['posicoes_pontos'])
    posicoes_demandas = np.array(visual['posicoes_demandas'])
    qualidades = visual['qualidades_pontos']
    demandas = visual['niveis_demanda']
    
    # Verificar se há solução
    tem_solucao = solucao is not None
    pontos_ativos = solucao.get('pontos_ativos', []) if tem_solucao else []
    
    # Calcular limites adaptativos
    x_min, x_max, y_min, y_max = calcular_limites_adaptativos(posicoes_pontos, padding_percent=0.12)
    
    # Ajustar para incluir demandas
    if len(posicoes_demandas) > 0:
        x_min_d, x_max_d, y_min_d, y_max_d = calcular_limites_adaptativos(posicoes_demandas, padding_percent=0)
        x_min = min(x_min, x_min_d)
        x_max = max(x_max, x_max_d)
        y_min = min(y_min, y_min_d)
        y_max = max(y_max, y_max_d)
        
        # Reaplicar padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * 0.08
        x_max += x_range * 0.08
        y_min -= y_range * 0.08
        y_max += y_range * 0.08
    
    # Criar figura
    aspect_ratio = (x_max - x_min) / (y_max - y_min) if (y_max - y_min) > 0 else 1
    fig_width = 14
    fig_height = fig_width / aspect_ratio if aspect_ratio > 0 else 10
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    
    # Configurar limites
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Desenhar grid
    desenhar_grid_adaptativo(ax, x_min, x_max, y_min, y_max)
    
    # Fundo
    ax.set_facecolor('#f8f9fa')
    
    # Tamanhos proporcionais
    range_x = x_max - x_min
    tamanho_base_ponto = max(80, min(300, range_x * 0.03))
    tamanho_base_demanda = max(30, min(150, range_x * 0.02))
    
    # === DESENHAR DEMANDAS ===
    max_demanda = max(demandas) if demandas else 1
    for i, (x, y) in enumerate(posicoes_demandas):
        tamanho = tamanho_base_demanda + (demandas[i] / max_demanda) * tamanho_base_demanda
        ax.scatter(x, y, s=tamanho, c='#87CEEB', alpha=0.7, 
                  edgecolors='#1E90FF', linewidth=1.5, zorder=2)
        
        if len(posicoes_demandas) <= 30:
            ax.annotate(f'D{i+1}', xy=(x, y), xytext=(5, 5),
                       textcoords='offset points', fontsize=8, ha='left', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
                       zorder=3)
    
    # === DESENHAR PONTOS CANDIDATOS ===
    for i, (x, y) in enumerate(posicoes_pontos):
        is_active = (i + 1) in pontos_ativos if tem_solucao else False
        qualidade = qualidades[i]
        
        if is_active and tem_solucao:
            cor_ponto = '#2E8B57'
            edgecolor = '#1a5a38'
            marker = 's'
            alpha = 0.95
        else:
            # Gradiente de qualidade (verde melhor, vermelho pior)
            r = max(0.2, 1.0 - qualidade)
            g = max(0.3, qualidade)
            b = 0.2
            cor_ponto = (r, g, b)
            edgecolor = '#333333'
            marker = 'o'
            alpha = 0.85
        
        tamanho = tamanho_base_ponto + qualidade * tamanho_base_ponto * 0.5
        
        ax.scatter(x, y, s=tamanho, c=[cor_ponto], marker=marker, 
                  edgecolors=edgecolor, linewidth=2, zorder=4, alpha=alpha)
        
        # ID do ponto
        ax.annotate(f'{i+1}', xy=(x, y), ha='center', va='center', 
                   fontsize=9, fontweight='bold', color='white', zorder=5)
        
        # Qualidade (se não houver muitos pontos)
        if len(posicoes_pontos) <= 40:
            ax.annotate(f'{qualidade:.2f}', xy=(x, y), xytext=(0, tamanho**0.5 * 0.6),
                       textcoords='offset points', fontsize=7, ha='center',
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8),
                       zorder=5)
    
    # === CÍRCULO DE ACESSIBILIDADE ===
    if len(posicoes_demandas) > 0:
        demanda_exemplo = posicoes_demandas[0]
        d_walk_max = params.get('d_walk_max', 500)
        
        circulo = Circle((demanda_exemplo[0], demanda_exemplo[1]), d_walk_max, 
                        fill=False, edgecolor='#1E90FF', linestyle='--', 
                        linewidth=1.5, alpha=0.4, zorder=1)
        ax.add_patch(circulo)
        
        ax.text(demanda_exemplo[0] + d_walk_max * 0.6, 
                demanda_exemplo[1] + d_walk_max * 0.6,
                f'Raio de acessibilidade\n{d_walk_max:.0f} m', 
                fontsize=8, alpha=0.6, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # === ESCALA E ROSA DOS VENTOS ===
    desenhar_barra_escala(ax, x_min, x_max, y_min, y_max)
    desenhar_rosa_dos_ventos(ax, x_min, x_max, y_min, y_max)
    
    # === TÍTULO E RÓTULOS ===
    ax.set_xlabel('Coordenada X (metros)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Coordenada Y (metros)', fontsize=11, fontweight='bold')
    
    if tem_solucao:
        titulo = f'SOLUÇÃO DO MODELO DE OTIMIZAÇÃO\n'
        titulo += f'{len(posicoes_pontos)} Pontos | {len(pontos_ativos)} Ativos | {len(posicoes_demandas)} Demandas'
        titulo += f'\nValor Objetivo: {solucao.get("valor_objetivo", 0):.2f}'
    else:
        titulo = f'DADOS DE ENTRADA (PRÉ-OTIMIZAÇÃO)\n'
        titulo += f'{len(posicoes_pontos)} Pontos Candidatos | {len(posicoes_demandas)} Nós de Demanda'
    ax.set_title(titulo, fontsize=14, fontweight='bold', pad=20)
    
    # === INFORMAÇÕES DO MODELO ===
    info_text = f"PARÂMETROS DO MODELO:\n"
    info_text += f"• d_walk_max = {params.get('d_walk_max', 500):.0f} m\n"
    info_text += f"• d_route_max = {params.get('d_route_max', 800):.0f} m\n"
    info_text += f"• Capt = {params.get('Capt', 800)}\n"
    info_text += f"• m_max = {params.get('m_max', 3)} rotas/ponto\n"
    info_text += f"• ω = {params.get('omega', 50)}\n"
    info_text += f"• Demanda total = {estatisticas.get('demanda_total', 0)}"
    
    ax.text(x_min, y_max, info_text, transform=ax.transData,
            fontsize=9, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='#FFFFE0', alpha=0.9, edgecolor='#CCCCCC'),
            zorder=10)
    
    # === LEGENDAS ===
    if tem_solucao:
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#87CEEB', 
                   markersize=10, label='Nós de Demanda', markeredgecolor='#1E90FF', markeredgewidth=1.5),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=10, label='Ponto (não ativo)', markeredgecolor='#333333'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#2E8B57', 
                   markersize=10, label='Ponto Ativo (selecionado)', markeredgecolor='#1a5a38'),
            Line2D([0], [0], linestyle='--', color='#1E90FF', linewidth=1.5, 
                   alpha=0.6, label=f'Raio de Acessibilidade ({params.get("d_walk_max", 500):.0f}m)'),
        ]
    else:
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#87CEEB', 
                   markersize=10, label='Nós de Demanda', markeredgecolor='#1E90FF', markeredgewidth=1.5),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=(0.6, 0.6, 0.2), 
                   markersize=10, label='Ponto Candidato (cor = qualidade)', markeredgecolor='#333333'),
            Line2D([0], [0], linestyle='--', color='#1E90FF', linewidth=1.5, 
                   alpha=0.6, label=f'Raio de Acessibilidade ({params.get("d_walk_max", 500):.0f}m)'),
        ]
        
        # Gradiente de cores
        from matplotlib.patches import Rectangle
        gradiente_ax = fig.add_axes([0.85, 0.02, 0.12, 0.03])
        gradiente_ax.set_xlim(0, 1)
        gradiente_ax.set_ylim(0, 1)
        for i in range(100):
            cor = (1 - i/100, i/100, 0.2)
            gradiente_ax.add_patch(Rectangle((i/100, 0), 0.01, 1, color=cor))
        gradiente_ax.set_xticks([0, 1])
        gradiente_ax.set_xticklabels(['Ruim', 'Bom'])
        gradiente_ax.set_yticks([])
        gradiente_ax.set_title('Qualidade', fontsize=7)
    
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9, 
             framealpha=0.9, edgecolor='#CCCCCC', ncol=1)
    
    plt.tight_layout()
    return fig, ax

def visualizar_densidade(dados_json):
    """Cria visualização de densidade dos pontos."""
    
    if dados_json is None:
        return None, None
    
    visual = dados_json['visualizacao']
    estatisticas = dados_json['estatisticas']
    params = dados_json['parametros']
    
    posicoes_pontos = np.array(visual['posicoes_pontos'])
    posicoes_demandas = np.array(visual['posicoes_demandas'])
    qualidades = visual['qualidades_pontos']
    demandas = visual['niveis_demanda']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Densidade dos pontos candidatos
    ax1 = axes[0, 0]
    hb1 = ax1.hexbin(posicoes_pontos[:, 0], posicoes_pontos[:, 1], gridsize=25, 
                     cmap='YlOrRd', alpha=0.7, mincnt=1)
    ax1.scatter(posicoes_pontos[:, 0], posicoes_pontos[:, 1], c='blue', s=20, alpha=0.5)
    ax1.set_title('Densidade dos Pontos Candidatos', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    plt.colorbar(hb1, ax=ax1, label='Densidade')
    
    # 2. Densidade das demandas
    ax2 = axes[0, 1]
    hb2 = ax2.hexbin(posicoes_demandas[:, 0], posicoes_demandas[:, 1], gridsize=25, 
                     cmap='Blues', alpha=0.7, mincnt=1)
    ax2.scatter(posicoes_demandas[:, 0], posicoes_demandas[:, 1], c='red', s=20, alpha=0.5)
    ax2.set_title('Densidade dos Nós de Demanda', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    plt.colorbar(hb2, ax=ax2, label='Densidade')
    
    # 3. Distribuição das qualidades
    ax3 = axes[1, 0]
    ax3.hist(qualidades, bins=20, color='green', alpha=0.7, edgecolor='black')
    ax3.set_title('Distribuição da Qualidade dos Pontos (w[n])', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Qualidade')
    ax3.set_ylabel('Frequência')
    ax3.axvline(estatisticas['qualidade_media'], color='red', linestyle='--', 
                label=f'Média: {estatisticas["qualidade_media"]:.3f}')
    ax3.legend()
    
    # 4. Distribuição das demandas
    ax4 = axes[1, 1]
    ax4.hist(demandas, bins=20, color='blue', alpha=0.7, edgecolor='black')
    ax4.set_title('Distribuição dos Níveis de Demanda (de[q])', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Demanda (passageiros)')
    ax4.set_ylabel('Frequência')
    ax4.axvline(estatisticas['demanda_media'], color='red', linestyle='--', 
                label=f'Média: {estatisticas["demanda_media"]:.1f}')
    ax4.legend()
    
    plt.suptitle(f'Análise Estatística dos Dados (semente: {dados_json["metadata"]["semente"]})', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig, axes

def visualizar_comparativo(dados_json, solucao):
    """Cria visualização comparativa antes/depois da otimização."""
    
    if dados_json is None or solucao is None:
        return None, None
    
    visual = dados_json['visualizacao']
    pontos_ativos = solucao.get('pontos_ativos', [])
    
    posicoes_pontos = np.array(visual['posicoes_pontos'])
    qualidades = visual['qualidades_pontos']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Gráfico 1: Antes (todos os pontos são candidatos)
    for i, (x, y) in enumerate(posicoes_pontos):
        qualidade = qualidades[i]
        cor = (max(0.2, 1.0 - qualidade), max(0.3, qualidade), 0.2)
        ax1.scatter(x, y, s=200, c=[cor], marker='o', edgecolors='#333333', linewidth=2, alpha=0.8)
        ax1.annotate(f'{i+1}', xy=(x, y), ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    ax1.set_title(f'ANTES: {len(posicoes_pontos)} Pontos Candidatos', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    
    # Gráfico 2: Depois (apenas pontos ativos)
    for i, (x, y) in enumerate(posicoes_pontos):
        is_active = (i + 1) in pontos_ativos
        if is_active:
            ax2.scatter(x, y, s=250, c=['#2E8B57'], marker='s', edgecolors='#1a5a38', linewidth=2.5, alpha=0.95)
            ax2.annotate(f'{i+1}', xy=(x, y), ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        else:
            ax2.scatter(x, y, s=100, c=['lightgray'], marker='o', edgecolors='gray', linewidth=1, alpha=0.5)
            ax2.annotate(f'{i+1}', xy=(x, y), ha='center', va='center', fontsize=7, color='gray')
    
    ax2.set_title(f'DEPOIS: {len(pontos_ativos)} Pontos Ativos Selecionados', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    
    # Ajustar limites
    x_min, x_max, y_min, y_max = calcular_limites_adaptativos(posicoes_pontos, 0.1)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    
    plt.suptitle(f'Comparação Antes/Depois da Otimização\nValor Objetivo: {solucao.get("valor_objetivo", 0):.2f}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig, (ax1, ax2)

# ============================================
# FUNÇÃO PRINCIPAL
# ============================================

def main():
    """Função principal."""
    
    print("="*60)
    print("VISUALIZADOR DO SISTEMA DE PARADAS DE ÔNIBUS")
    print("="*60)
    
    # Carregar dados
    print("\nCarregando 'cenario_01.json'...")
    dados_json = ler_dados_json('cenario_02.json')
    
    if dados_json is None:
        return
    
    # Carregar solução (se existir)
    solucao = ler_solucao_json('solucao.json')
    
    # Exibir informações
    params = dados_json['parametros']
    estatisticas = dados_json['estatisticas']
    
    print(f"\nDados carregados:")
    print(f"  - Pontos candidatos: {params['NumN']}")
    print(f"  - Nós de demanda: {params['NumQ']}")
    print(f"  - Demanda total: {estatisticas['demanda_total']} passageiros")
    print(f"  - Qualidade média dos pontos: {estatisticas['qualidade_media']:.3f}")
    print(f"  - d_walk_max: {params['d_walk_max']} m")
    
    if solucao:
        print(f"\nSolução encontrada:")
        print(f"  - Valor objetivo: {solucao.get('valor_objetivo', 0):.2f}")
        print(f"  - Pontos ativos: {len(solucao.get('pontos_ativos', []))}")
    
    # Criar visualizações
    print("\nGerando visualização principal...")
    fig1, ax1 = visualizar_pontos(dados_json, solucao)
    
    print("Gerando análise de densidade...")
    fig2, axes2 = visualizar_densidade(dados_json)
    
    if solucao:
        print("Gerando visualização comparativa...")
        fig3, axes3 = visualizar_comparativo(dados_json, solucao)
    
    # Mostrar
    plt.show()
    
    # Salvar figuras
    try:
        salvar = input("\nDeseja salvar as figuras? (s/n): ").lower()
        if salvar == 's':
            fig1.savefig('visualizacao_pontos.png', dpi=150, bbox_inches='tight', facecolor='white')
            fig2.savefig('visualizacao_densidade.png', dpi=150, bbox_inches='tight', facecolor='white')
            if solucao:
                fig3.savefig('visualizacao_comparativo.png', dpi=150, bbox_inches='tight', facecolor='white')
            print("\nFiguras salvas:")
            print("  - visualizacao_pontos.png")
            print("  - visualizacao_densidade.png")
            if solucao:
                print("  - visualizacao_comparativo.png")
    except Exception as e:
        print(f"\nErro ao salvar: {e}")
    
    print("\n" + "="*60)
    print("Visualização concluída!")
    print("="*60)

if __name__ == "__main__":
    main()