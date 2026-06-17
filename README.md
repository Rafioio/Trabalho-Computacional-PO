# Trabalho-Computacional-PO

# Projeto SouBuz: Otimização de Pontos de Parada de Ônibus

Este projeto aplica técnicas de **Pesquisa Operacional (PO)** para resolver a ineficiência e o desequilíbrio espacial na alocação de pontos de ônibus em redes de transporte coletivo urbano. O modelo foi originalmente desenvolvido em OPL/CPLEX e migrado para Python com **Gurobi** como solver.

## 📌 O Problema: Desertos de Mobilidade

Em grandes centros urbanos, como Belo Horizonte, existe um conflito direto (trade-off) na distribuição de pontos de parada. Por um lado, pontos excessivamente próximos (ex: a cada 200m) reduzem a velocidade comercial dos ônibus e aumentam o tempo de viagem. Por outro lado, o espaçamento excessivo (ex: a cada 2km) cria verdadeiros "desertos de mobilidade", forçando os passageiros a realizarem caminhadas exaustivas que ultrapassam o limite de aceitabilidade.

## 🎯 Objetivo do Modelo

O modelo computacional foi desenvolvido para encontrar a distribuição ótima de abrigos de ônibus, equilibrando o acesso universal da população e a eficiência operacional da frota. Trata-se de um problema multi-objetivo convertido em uma função escalar através de pesos (`W1, W2, W3, W4`). O modelo visa:

1. **Minimizar o custo social (f1):** Reduzir a distância total caminhada pelos usuários e aplicar penalidades rigorosas caso a demanda de um bairro fique desassistida.
2. **Maximizar a viabilidade técnica (f2):** Priorizar a instalação de pontos em locais adequados de infraestrutura viária.
3. **Minimizar os custos de implantação (f3):** Reduzir o número total de paradas físicas construídas na cidade e o custo de capacidade adicional da frota.
4. **Minimizar a penalidade de espaçamento (f4):** Penalizar violações do espaçamento máximo entre paradas consecutivas.

## ⚙️ Características Técnicas e Modelagem

O modelo foi implementado em **Python** utilizando a API **Gurobi** (`gurobipy`), traduzido da formulação original em OPL/CPLEX.

- **Matrizes Esparsas para Otimização de Memória:** O código utiliza domínios filtrados para gerar as variáveis de decisão de embarque ($a_{qnk}$) apenas para distâncias lógicas ($d_{qn} \le d_{max\_walk}$), impedindo o solver de alocar processamento para passageiros a quilômetros de distância.
- **Controle de Espaçamento:** Implementa restrições sequenciais iterando sobre os vetores de rota para garantir que o espaçamento máximo (`d_route_max`) entre dois pontos consecutivos não seja violado na via.
- **Lotação e Fila de Ônibus:** Restrições limitam a capacidade máxima dos veículos e o número máximo de rotas que podem compartilhar o mesmo abrigo simultaneamente, prevenindo gargalos físicos.
- **Normalização de Pesos:** Implementa normalização utopia/anti-utopia para garantir que os diferentes objetivos (com escalas distintas) sejam combinados de forma justa.

## 🛠️ Pré-requisitos

Para rodar este projeto localmente, você precisará de:

- **Python 3.10+**
- **Gurobi Optimizer** com licença válida (`gurobipy`)
- Editor de código (ex: **VS Code**) com terminal integrado

## 📦 Instalação

Clone o repositório e instale as dependências:

```bash
git clone [https://github.com/seu-usuario/Trabalho-Computacional-PO.git](https://github.com/seu-usuario/Trabalho-Computacional-PO.git)
cd Trabalho-Computacional-PO

# Criar e ativar ambiente virtual
python3 -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows (PowerShell)

# Instalar dependências
pip install -r requirements.txt

```

**Arquivo `requirements.txt`:**

```text
gurobipy>=11.0
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
openpyxl>=3.1
pytest>=7.0
scipy>=1.10

```

## 📁 Estrutura de Arquivos

```text
src/
├── data/
│   ├── dados.dat          # Dados de entrada (formato OPL legado)
│   ├── dados.json         # Dados convertidos para JSON (carga rápida)
│   └── loader.py          # Parsing de .dat e I/O de .json
├── model/
│   ├── domains.py         # Construção dos domínios esparsos
│   ├── variables.py       # Definição das variáveis de decisão
│   ├── objective.py       # Função objetivo (f1-f4)
│   ├── constraints.py     # Restrições do modelo (8 grupos)
│   ├── solver.py          # Montagem e otimização
│   └── function_normalizer.py # Normalização Min-Max utopia/anti-utopia
├── scripts/
│   └── map_viewer.py      # Visualização de cenários e soluções
├── utils/
│   ├── generate_data.py      # Gerador de dados sintéticos realistas
│   ├── validator.py          # Validação de consistência dos dados
│   ├── export_solution.py    # Exportação de resultados (→ solucao.json)
│   ├── convert_csv_to_data.py# Conversor de dados BH (CSV → JSON)
│   └── reporting/            # Geração de relatórios e fronteira de Pareto
│       ├── run_pareto.py              # Experimento da fronteira de Pareto
│       ├── analyze_and_report.py      # Relatório completo dos experimentos
│       ├── generate_viz.py            # Geração de visualizações
│       └── run_remaining_and_report.py# Relatório dos tiers superiores
└── run.py                 # Ponto de entrada principal

```

## 🚀 Como Executar

### 1. Gerar Dados Sintéticos

Para testar o modelo sem dados reais, utilize o gerador de dados sintéticos:

```bash
# Gerar dados com configuração padrão (200 nós, 2 rotas, 70 zonas)
python src/utils/generate_data.py

# Gerar dados personalizados
python src/utils/generate_data.py --num-n 30 --num-k 3 --num-q 20

# Gerar dados com semente fixa (reprodutível)
python src/utils/generate_data.py --seed 42 --output meus_dados.json

# Escolher método de geração de rotas (nearest, grid, hybrid)
python src/utils/generate_data.py --route-method hybrid

# Gerar múltiplos cenários para teste
python src/utils/generate_data.py --scenarios 5 --prefix cenario

```

### 2. Executar a Otimização

```bash
# Executar com auto-detecção do arquivo de dados
python -m src.run

# Executar com arquivo específico
python -m src.run --data dados_generated.json

# Definir arquivo de saída da solução
python -m src.run --data dados_generated.json --output minha_solucao.json

```

### 3. Visualizar Resultados

```bash
# Visualizar dados de entrada (pré-otimização)
python src/scripts/map_viewer.py dados_generated.json

# Visualizar solução (pontos ativos + rotas)
python src/scripts/map_viewer.py dados_generated.json --solution solucao.json

# Visualizar apenas análise de densidade
python src/scripts/map_viewer.py dados_generated.json --density-only

# Visualizar comparação antes/depois
python src/scripts/map_viewer.py dados_generated.json --solution solucao.json --comparison

# Salvar figura em arquivo
python src/scripts/map_viewer.py dados_generated.json --solution solucao.json --output figura.png

```

### 4. Converter Dados da BH (CSV da PBH)

Converter o CSV público de pontos de ônibus de Belo Horizonte para o formato JSON do modelo:

```bash
# Converter todas as rotas (~70k pontos)
python src/utils/convert_csv_to_data.py --output dados_bh.json

# Instância pequena para testes (3 rotas, 20 pontos cada, 5 zonas)
python src/utils/convert_csv_to_data.py \
    --max-routes 3 --max-stops 20 --num-q 5 \
    --output small_bh.json

# Instância personalizada
python src/utils/convert_csv_to_data.py \
    --max-routes 5 --max-stops 50 --num-q 30 \
    --d-walk-max 400 --d-route-max 800 \
    --capt 1000 --seed 123 \
    --output dados_bh_5rotas.json
```

O CSV de entrada deve estar em `src/data/20260504_ponto_onibus.csv`. O conversor:
- Deduplica pontos pelo par de coordenadas UTM
- Preserva a ordem das paradas por rota
- Gera zonas de demanda sintéticas próximas aos pontos reais
- Atribui índices de qualidade técnica via distribuição beta

### 5. Fronteira de Pareto

Varrer o macro-peso mu com granularidade fina para construir a fronteira de Pareto:

```bash
# Executar experimento Pareto (tier pequena: N=50, K=4, Q=10)
python src/utils/reporting/run_pareto.py
```

O script:
- Gera dados sintéticos para as sementes 42 e 123
- Aplica normalização utopia/anti-utopia
- Varre mu de 0.1 a 3.0 (passo 0.1) com theta=1.0 fixo
- Constrói a fronteira de Pareto (50 pontos não-dominados)
- Gera gráfico em `docs/pareto/pareto_frontier.png`
- Salva relatório em `docs/pareto/pareto_report.md`

### 6. Relatório Completo dos Experimentos

Gerar relatório consolidado a partir dos resultados do batch_runner:

```bash
# Analisar runs e gerar relatório com visualizações
python src/utils/reporting/analyze_and_report.py

# Executar tiers não completados (muito_grande, extrema) e gerar relatório
python src/utils/reporting/run_remaining_and_report.py

# Regenerar visualizações para runs selecionadas
python src/utils/reporting/generate_viz.py
```

### 7. Exportar Solução para Diferentes Formatos

```bash
# Exportar solução para JSON (já feito automaticamente pelo run.py)
python -c "from src.utils.export_solution import export_solution; from src.model.solver import build_and_solve; from src.data.loader import load_json; d=load_json('dados_generated.json'); r=build_and_solve(d); export_solution(r, d, path='solucao.json')"

# Exportar para Excel/CSV
python -c "from src.utils.export_solution import export_solution_csv; from src.model.solver import build_and_solve; from src.data.loader import load_json; d=load_json('dados_generated.json'); r=build_and_solve(d); export_solution_csv(r, d, path='solucao.xlsx')"

```

## 📊 Exemplo de Fluxo Completo

```bash
# 1. Gerar dados sintéticos
python src/utils/generate_data.py --num-n 30 --num-k 3 --num-q 20 --seed 123

# 2. Executar otimização
python -m src.run --data dados_generated.json -v

# 3. Visualizar solução
python src/scripts/map_viewer.py dados_generated.json --solution solucao.json --output visualizacao.png

```

## 🔧 Parâmetros do Modelo

### Parâmetros do Gerador de Dados

| Parâmetro | Descrição | Padrão |
| --- | --- | --- |
| `--num-n` | Número de pontos candidatos | 200 |
| `--num-k` | Número de rotas | 2 |
| `--num-q` | Número de zonas de demanda | 70 |
| `--route-len` | Nós por rota | ~35% de NumN |
| `--grid-width/height` | Dimensões da cidade (m) | 3000 x 3000 |
| `--min-dist` | Distância mínima entre pontos (m) | 80 |
| `--d-walk-max` | Distância máxima de caminhada (m) | 400 |
| `--d-route-max` | Distância máxima entre paradas (m) | 800 |
| `--capt` | Capacidade base do sistema | 800 |
| `--m-max` | Máximo de rotas por ponto | 7 |
| `--route-method` | Método de geração de rotas | hybrid |

### Pesos da Função Objetivo

| Parâmetro | Descrição | Padrão |
| --- | --- | --- |
| `--W1` | Peso do custo social (f1) | 1 |
| `--W2` | Peso da penalidade de espaçamento (f2) | 1 |
| `--W3` | Peso do custo de infraestrutura (f3) | 1 |
| `--W4` | Peso da viabilidade técnica (f4) | 1 |

> ⚠️ **Observação:** A normalização Min-Max é aplicada automaticamente via matriz payoff utopia/anti-utopia antes da otimização, tornando as funções comparáveis independentemente da escala.

## 📈 Interpretação dos Resultados

### Função Objetivo

* **f1 (custo social):** Distância total caminhada + penalidades por demanda não atendida
* **f2 (penalidade de espaçamento):** Soma das folgas (slack) para violações do espaçamento máximo entre paradas consecutivas
* **f3 (custo infraestrutura):** Número de pontos ativos + custo de capacidade adicional
* **f4 (viabilidade técnica):** Soma do inverso da qualidade dos pontos ativos ponderada pela capacidade alocada

### Visualização

* **Pontos verdes (quadrados):** Pontos de parada ativos (selecionados pela otimização)
* **Pontos cinza (círculos):** Pontos candidatos não ativos
* **Círculos azuis:** Nós de demanda (tamanho proporcional à demanda)
* **Linhas coloridas:** Rotas (cada rota tem uma cor única)
* **Círculo tracejado:** Raio de acessibilidade (d_walk_max)

## 🔄 Histórico de Migração

Este repositório foi originalmente implementado em **OPL/CPLEX**. O modelo foi traduzido para **Python/Gurobi** e modularizado para melhor manutenibilidade. A implementação original em OPL foi removida após a conclusão da migração.

## 📝 Licença

Este projeto é de uso acadêmico para a disciplina de Pesquisa Operacional.

## 👥 Autores

* Desenvolvido como Trabalho Computacional de PO