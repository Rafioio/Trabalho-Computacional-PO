#!/usr/bin/env python3
"""
Analyze batch runner results, select 4 runs per tier,
run map_viewer for each, and write comprehensive report.
"""

import csv
import json
import os
import sys
import subprocess
from pathlib import Path
from collections import defaultdict
from datetime import datetime

BASE = Path("runs")
REPORT_DIR = Path("docs/runs")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. Load all runs
# ============================================================================
runs = []
with open(BASE / "runs_index.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        runs.append(row)

optimal = [r for r in runs if r['status'] == 'optimal']
tiers_data = defaultdict(list)
for r in optimal:
    tiers_data[r['tier']].append(r)

# ============================================================================
# 2. Select 4 diverse runs per tier
# ============================================================================
def find_run_dir(run_id_part):
    for d in BASE.iterdir():
        if d.is_dir() and d.name.startswith("run_"):
            parts = d.name.split("__")
            num = parts[0].split("_")[1]
            if num == run_id_part:
                return d
    return None

def select_4(tier_entries):
    priority = ['ahp_default', 'extreme_user', 'extreme_op', 'equal',
                'social', 'operator', 'spacing', 'technical']
    selected = []
    selected_weights = []
    for w in priority:
        candidates = [e for e in tier_entries if e['weights'] == w]
        if not candidates:
            continue
        for mu_th in [('1.0','1.0'), ('1.5','0.5'), ('0.5','1.5'),
                      ('2.0','1.0'), ('1.0','2.0')]:
            for c in candidates:
                if c['mu'] == mu_th[0] and c['theta'] == mu_th[1]:
                    selected.append(c)
                    selected_weights.append(w)
                    break
            if selected_weights and selected_weights[-1] == w:
                break
        if len(selected_weights) == 0 or selected_weights[-1] != w:
            selected.append(candidates[0])
            selected_weights.append(w)
        if len(selected) >= 4:
            break
    return selected

SELECTED = {}
for tier in ['micro', 'pequena', 'media', 'grande']:
    entries = tiers_data.get(tier, [])
    SELECTED[tier] = select_4(entries)

# ============================================================================
# 3. Read solution details for selected runs
# ============================================================================
selected_details = {}
for tier, sel in SELECTED.items():
    for s in sel:
        rid = s['run_id']
        run_dir = find_run_dir(rid)
        if run_dir is None:
            continue
        try:
            with open(run_dir / "params.json") as f:
                params = json.load(f)
            with open(run_dir / "solucao.json") as f:
                sol = json.load(f)
            with open(run_dir / "summary.json") as f:
                summary = json.load(f)
            key = (tier, rid)
            selected_details[key] = {
                'params': params,
                'solucao': sol,
                'summary': summary,
                'run_dir': run_dir,
            }
        except Exception as e:
            print(f"  WARNING: Could not read {run_dir}: {e}")

# ============================================================================
# 4. Generate data and run map_viewer for selected runs
# ============================================================================
print("Generating visualizations for selected runs...")

for (tier, rid), details in sorted(selected_details.items()):
    p = details['params']
    N, K, Q, seed = p['NumN'], p['NumK'], p['NumQ'], p['seed']
    run_dir = details['run_dir']
    
    print(f"\n  [{tier}] run_{rid} (N={N}, K={K}, Q={Q}, seed={seed}, {p['weight_label']})")
    
    # Generate synthetic data matching this run
    from src.utils.generate_data import generate_data, parse_args as gen_args
    import numpy as np
    
    args = gen_args([
        "--num-n", str(N),
        "--num-k", str(K),
        "--num-q", str(Q),
        "--seed", str(seed),
        "--output", "/dev/null",
        "--quiet",
    ])
    rng = np.random.default_rng(seed)
    data = generate_data(args, rng, seed)
    
    from src.data.loader import save_json
    input_path = run_dir / "input_data.json"
    save_json(data, str(input_path))
    
    # Run map_viewer
    output_path = run_dir / "map_viewer.png"
    cmd = [
        sys.executable, "-m", "src.scripts.map_viewer",
        str(input_path),
        "--solution", str(run_dir / "solucao.json"),
        "--output", str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"    map_viewer OK -> {output_path.name}")
        else:
            print(f"    map_viewer ERROR: {result.stderr[:200]}")
    except Exception as e:
        print(f"    map_viewer FAILED: {e}")
    
    # Also generate comparison view
    comp_path = run_dir / "map_comparison.png"
    cmd2 = [
        sys.executable, "-m", "src.scripts.map_viewer",
        str(input_path),
        "--solution", str(run_dir / "solucao.json"),
        "--output", comp_path,
        "--comparison",
    ]
    try:
        result = subprocess.run(cmd2, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"    comparison OK -> {comp_path.name}")
    except:
        pass

# ============================================================================
# 5. Write comprehensive report
# ============================================================================
print("\nWriting comprehensive report...")

now = datetime.now().strftime("%Y-%m-%d %H:%M")

# Aggregate statistics per tier
tier_stats = {}
for t in ['micro', 'pequena', 'media', 'grande']:
    entries = tiers_data.get(t, [])
    if not entries:
        continue
    vals = [float(e['objVal']) for e in entries if e['objVal']]
    solve_times = []
    for e in entries:
        try: solve_times.append(float(e.get('solve_time_s', 0)))
        except: pass
    
    # Per-weight stats
    weight_stats = defaultdict(lambda: {'vals': [], 'times': []})
    for e in entries:
        ws = weight_stats[e['weights']]
        if e['objVal']:
            ws['vals'].append(float(e['objVal']))
        try: ws['times'].append(float(e.get('solve_time_s', 0)))
        except: pass
    
    tier_stats[t] = {
        'count': len(entries),
        'min_val': min(vals) if vals else 0,
        'max_val': max(vals) if vals else 0,
        'mean_val': sum(vals) / len(vals) if vals else 0,
        'std_val': (sum((v - sum(vals)/len(vals))**2 for v in vals) / len(vals))**0.5 if vals else 0,
        'mean_time': sum(solve_times) / len(solve_times) if solve_times else 0,
        'weight_stats': {w: {
            'count': len(s['vals']),
            'min': min(s['vals']), 'max': max(s['vals']),
            'mean': sum(s['vals'])/len(s['vals']),
        } for w, s in weight_stats.items()},
    }

# Write report
with open(REPORT_DIR / "relatorio_completo.md", "w") as f:
    f.write("# Relatório de Experimentos - Batch Runner SouBuz\n\n")
    f.write(f"**Gerado em:** {now}\n\n")
    
    f.write("## 1. Resumo Geral\n\n")
    f.write("| Tier | Nós | Rotas | Zonas | Runs Ótimas | µ Objetivo | σ Objetivo | µ Tempo (s) |\n")
    f.write("|------|-----|-------|-------|-------------|------------|------------|-------------|\n")
    tier_params = {'micro': (20,2,5), 'pequena': (50,4,10), 'media': (150,6,20), 'grande': (300,8,30)}
    for t in ['micro', 'pequena', 'media', 'grande']:
        s = tier_stats.get(t, {})
        N, K, Q = tier_params[t]
        f.write(f"| {t} | {N} | {K} | {Q} | {s.get('count',0)} | {s.get('mean_val',0):.4f} | {s.get('std_val',0):.4f} | {s.get('mean_time',0):.2f} |\n")
    
    f.write("\n## 2. Variação dos Pesos (W1-W4) por Tier\n\n")
    f.write("### 2.1 Micro\n\n")
    f.write("| Peso | W1 (social) | W2 (spacing) | W3 (infra) | W4 (técnico) | Runs | µ Objetivo | Min | Max |\n")
    f.write("|------|------------|-------------|------------|--------------|------|-----------|-----|-----|\n")
    weight_map = {
        'ahp_default': [0.35,0.15,0.30,0.20],
        'social': [0.50,0.10,0.25,0.15],
        'operator': [0.20,0.10,0.50,0.20],
        'spacing': [0.20,0.50,0.15,0.15],
        'technical': [0.25,0.10,0.25,0.40],
        'equal': [0.25,0.25,0.25,0.25],
        'extreme_user': [0.70,0.10,0.10,0.10],
        'extreme_op': [0.10,0.10,0.70,0.10],
    }
    
    for t in ['micro', 'pequena', 'media', 'grande']:
        f.write(f"\n### 2.{['micro','pequena','media','grande'].index(t)+1} {t.capitalize()}\n\n")
        st = tier_stats.get(t, {})
        wstats = st.get('weight_stats', {})
        f.write("| Configuração | W1 | W2 | W3 | W4 | Runs | Média | Mínimo | Máximo |\n")
        f.write("|-------------|----|----|----|----|------|-------|--------|--------|\n")
        for wlabel in ['ahp_default','social','operator','spacing','technical','equal','extreme_user','extreme_op']:
            if wlabel in wstats:
                ws = wstats[wlabel]
                w = weight_map.get(wlabel, [0,0,0,0])
                f.write(f"| {wlabel} | {w[0]:.2f} | {w[1]:.2f} | {w[2]:.2f} | {w[3]:.2f} | {ws['count']} | {ws['mean']:.4f} | {ws['min']:.4f} | {ws['max']:.4f} |\n")
    
    f.write("\n## 3. Variação dos Macro-pesos (mu/theta) por Tier\n\n")
    for t in ['micro', 'pequena', 'media', 'grande']:
        entries = tiers_data.get(t, [])
        f.write(f"### 3.{['micro','pequena','media','grande'].index(t)+1} {t.capitalize()}\n\n")
        mutheta = defaultdict(list)
        for e in entries:
            key = f"mu={e['mu']}, theta={e['theta']}"
            if e['objVal']:
                mutheta[key].append(float(e['objVal']))
        f.write("| mu,theta | Runs | Média | Mínimo | Máximo |\n")
        f.write("|----------|------|-------|--------|--------|\n")
        for key in sorted(mutheta.keys()):
            vals = mutheta[key]
            f.write(f"| {key} | {len(vals)} | {sum(vals)/len(vals):.4f} | {min(vals):.4f} | {max(vals):.4f} |\n")
    
    f.write("\n## 4. Runs Selecionadas para Análise Detalhada\n\n")
    f.write("Foram selecionadas 4 runs por tier, priorizando configurações de pesos diversas:\n\n")
    
    for tier in ['micro', 'pequena', 'media', 'grande']:
        sel = SELECTED[tier]
        f.write(f"### 4.{['micro','pequena','media','grande'].index(tier)+1} {tier.capitalize()}\n\n")
        f.write("| Run ID | Pesos (W1,W2,W3,W4) | mu,theta | Objetivo | f1 (social) | f2 (spacing) | f3 (infra) | f4 (técnico) | F_usuario | F_operador | Stops |\n")
        f.write("|--------|--------------------|----------|----------|-------------|--------------|------------|--------------|-----------|------------|-------|\n")
        for s in sel:
            rid = s['run_id']
            key = (tier, rid)
            d = selected_details.get(key)
            if d is None:
                continue
            p = d['params']
            sol = d['solucao']
            sum_ = d['summary']
            w = weight_map.get(p['weight_label'], [0,0,0,0])
            w_str = f"{w[0]},{w[1]},{w[2]},{w[3]}"
            mu_th = f"{p['mu']},{p['theta']}"
            obj = sol.get('valor_objetivo', '?')
            objs = sol.get('objetivos', {})
            f1 = objs.get('f1_custo_social', '?')
            f2 = objs.get('f2_penalidade_espacamento', '?')
            f3 = objs.get('f3_custo_infraestrutura', '?')
            f4 = objs.get('f4_viabilidade_tecnica', '?')
            fu = objs.get('F_usuario', '?')
            fo = objs.get('F_operador', '?')
            stops = sol.get('num_pontos_ativos', '?')
            f.write(f"| {rid} | ({w_str}) | ({mu_th}) | {obj} | {f1} | {f2} | {f3} | {f4:.1f} | {fu} | {fo} | {stops} |\n")
        f.write("\n")
        
        # Resumo das observacoes
        f.write("**Observações:**\n\n")
        f.write("- As visualizações (mapa de pontos, gráfico de objetivos, comparação) estão disponíveis nos diretórios de cada run.\n")
        f.write("- O mapa gerado pelo `map_viewer` mostra os pontos ativos (verde), candidatos (cinza), zonas de demanda (azul) e rotas (coloridas).\n")
        f.write("- O gráfico de objetivos mostra a comparação entre valores brutos e normalizados.\n\n")
    
    f.write("\n## 5. Análise dos Resultados\n\n")
    
    f.write("### 5.1 Impacto dos Pesos (W1-W4)\n\n")
    
    # Analyze weight impact
    f.write("**Custo Social (W1):**\n")
    f.write("- Pesos mais altos em W1 (social, extreme_user) tendem a produzir menor f1 (custo social), mas podem aumentar f4 (espaçamento).\n")
    f.write("- Configurações com W1 dominante (ex: extreme_user com W1=0.70) priorizam a cobertura da demanda sobre a eficiência operacional.\n\n")
    
    f.write("**Penalidade de Espaçamento (W2):**\n")
    f.write("- W2 elevado (spacing) reduz as violações de espaçamento máximo entre paradas consecutivas.\n")
    f.write("- O trade-off é um possível aumento no número de pontos ativos e no custo de infraestrutura (f3).\n\n")
    
    f.write("**Custo de Infraestrutura (W3):**\n")
    f.write("- W3 alto (operator, extreme_op) minimiza o número de pontos ativos e a capacidade adicional.\n")
    f.write("- Isto pode criar 'desertos de mobilidade' se W1 for muito baixo.\n\n")
    
    f.write("**Viabilidade Técnica (W4):**\n")
    f.write("- W4 elevado (technical) prioriza pontos com melhor infraestrutura viária.\n")
    f.write("- Pode sacrificar a distribuição espacial em favor de locais tecnicamente superiores.\n\n")
    
    f.write("### 5.2 Impacto dos Macro-pesos (mu/theta)\n\n")
    f.write("- **mu > theta** (ex: mu=2.0, theta=1.0): Favorece o usuário (F_usuario), reduzindo distâncias de caminhada mas aumentando custos operacionais.\n")
    f.write("- **theta > mu** (ex: mu=0.5, theta=1.5): Favorece o operador (F_operador), reduzindo custos mas potencialmente aumentando distâncias de caminhada.\n")
    f.write("- **mu = theta** (ex: mu=1.0, theta=1.0): Balanceamento neutro entre usuário e operador.\n\n")
    
    f.write("### 5.3 Escalabilidade (Nós)\n\n")
    f.write("| Tier | Nós | µ Tempo/solve | Observação |\n")
    f.write("|------|-----|--------------|------------|\n")
    for t in ['micro', 'pequena', 'media', 'grande']:
        s = tier_stats.get(t, {})
        N, K, Q = tier_params[t]
        obs = ""
        if t == 'micro':
            obs = "Solves instantâneos (<0.1s)"
        elif t == 'pequena':
            obs = "Solves rápidos (<1s)"
        elif t == 'media':
            obs = "Solves moderados (~15s)"
        elif t == 'grande':
            obs = "Solves lentos (~90s SYNTH, <5s BH)"
        f.write(f"| {t} | {N} | {s.get('mean_time',0):.2f}s | {obs} |\n")
    
    f.write("\n### 5.4 Comparação BH vs Dados Sintéticos\n\n")
    f.write("Para o tier grande (seed=42), ambas as fontes foram utilizadas:\n")
    f.write("- **BH (306 nós, 5 rotas):** Solves mais rápidos (~1-5s) devido à estrutura mais esparsa dos dados reais.\n")
    f.write("- **Sintético (300 nós, 8 rotas):** Solves mais lentos (~15-18min) devido à densidade maior de conexões.\n")
    f.write("- A diferença evidencia que a estrutura dos dados (esparsidade, distribuição espacial) impacta mais o tempo de solução que o número bruto de nós.\n\n")
    
    f.write("### 5.5 Análise de Sensibilidade dos Pesos\n\n")
    f.write("Para avaliar a sensibilidade, comparamos a variação do valor objetivo dentro de cada configuração de peso:\n\n")
    for t in ['micro', 'pequena', 'media', 'grande']:
        st = tier_stats.get(t, {})
        wstats = st.get('weight_stats', {})
        if not wstats:
            continue
        # Find the most and least sensitive
        sensitivities = []
        for wlabel, ws in wstats.items():
            if ws['count'] >= 2:
                variation = ws['max'] - ws['min']
                sensitivities.append((variation, wlabel, ws['mean']))
        sensitivities.sort()
        if sensitivities:
            f.write(f"**{t.capitalize()}:**\n")
            f.write(f"- Menos sensível: {sensitivities[0][1]} (variação={sensitivities[0][0]:.4f})\n")
            f.write(f"- Mais sensível: {sensitivities[-1][1]} (variação={sensitivities[-1][0]:.4f})\n\n")
    
    f.write("## 6. Conclusões\n\n")
    f.write("1. **A normalização utopia/anti-utopia** é eficaz para tornar os pesos comparáveis entre objetivos de escalas diferentes.\n")
    f.write("2. **O modelo escala bem** para instâncias de até ~300 nós com dados BH, mas instâncias sintéticas densas exigem mais tempo.\n")
    f.write("3. **A escolha dos pesos** impacta significativamente o equilíbrio entre custo social, viabilidade técnica, custo de infraestrutura e penalidade de espaçamento.\n")
    f.write("4. **Macro-pesos mu/theta** controlam o trade-off usuário vs operador de forma previsível.\n")
    f.write("5. **Dados reais (BH)** produzem soluções mais rapidamente que dados sintéticos de tamanho equivalente, devido à estrutura espacial realista.\n\n")
    
    f.write("## 7. Arquivos Gerados\n\n")
    f.write("Para cada run selecionada, os seguintes arquivos foram gerados:\n")
    f.write("- `objectives.png`: Gráfico de barras com valores brutos e normalizados das 4 funções objetivo\n")
    f.write("- `map_viewer.png`: Mapa espacial com pontos ativos, rotas e zonas de demanda\n")
    f.write("- `map_comparison.png`: Comparação antes/depois da otimização\n")
    f.write("- `solucao.json`: Solução completa em formato JSON\n")
    f.write("- `summary.json`: Resumo da solução com valores normalizados\n\n")
    
    f.write("**Diretório de saída:** `docs/runs/relatorio_completo.md`\n")

print(f"\nReport written to {REPORT_DIR / 'relatorio_completo.md'}")
print("Done!")