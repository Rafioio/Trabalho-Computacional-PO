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
- **Normalização de Pesos:** Implementa normalização utopia/anti-utopia para garantir que os diferentes objetivos (com escalas distintas) sejam combinados de forma franca.

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