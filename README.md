# Trabalho-Computacional-PO

# Projeto SouBuz: Otimização de Pontos de Parada de Ônibus

Este projeto aplica técnicas de **Pesquisa Operacional (PO)** para resolver a ineficiência e o desequilíbrio espacial na alocação de pontos de ônibus em redes de transporte coletivo urbano. 

## 📌 O Problema: Desertos de Mobilidade
Em grandes centros urbanos, como Belo Horizonte, existe um conflito direto (trade-off) na distribuição de pontos de parada. Por um lado, pontos excessivamente próximos (ex: a cada 200m) reduzem a velocidade comercial dos ônibus e aumentam o tempo de viagem. Por outro lado, o espaçamento excessivo (ex: a cada 2km) cria verdadeiros "desertos de mobilidade", forçando os passageiros a realizarem caminhadas exaustivas que ultrapassam o limite de aceitabilidade.

## 🎯 Objetivo do Modelo
O modelo computacional foi desenvolvido para encontrar a distribuição ótima de abrigos de ônibus, equilibrando o acesso universal da população e a eficiência operacional da frota. Trata-se de um problema multi-objetivo convertido em uma função escalar através de pesos ($W_1, W_2, W_3$) [1]. O modelo visa:
1. **Minimizar o custo social ($f_1$):** Reduzir a distância total caminhada pelos usuários e aplicar penalidades rigorosas caso a demanda de um bairro fique desassistida [2].
2. **Maximizar a viabilidade técnica ($f_2$):** Priorizar a instalação de pontos em locais adequados de infraestrutura viária [2].
3. **Minimizar os custos de implantação ($f_3$):** Reduzir o número total de paradas físicas construídas na cidade [2].

## ⚙️ Características Técnicas e Modelagem
A modelagem foi traduzida para **OPL (Optimization Programming Language)**, a linguagem nativa do solver IBM ILOG CPLEX [1].

* **Matrizes Esparsas para Otimização de Memória:** O código utiliza conjuntos filtrados (tuplas e `setof`) para gerar as variáveis de decisão de embarque ($a_{qnk}$) apenas para distâncias lógicas ($d_{qn} \le d_{max}^{walk}$), impedindo o solver de alocar processamento para passageiros a quilômetros de distância [3].
* **Controle de Espaçamento:** Implementa restrições sequenciais iterando sobre os vetores de rota para garantir que o espaçamento máximo ($droutemax$) entre dois pontos consecutivos não seja violado na via [3, 4].
* **Lotação e Fila de Ônibus:** Restrições limitam a capacidade máxima dos veículos e o número máximo de rotas que podem compartilhar o mesmo abrigo simultaneamente, prevenindo gargalos físicos [5-7].

## 🛠️ Pré-requisitos
Para rodar este projeto localmente, você precisará de:
* **IBM ILOG CPLEX Optimization Studio** (que fornece o compilador OPL).
* Editor de código (ex: **VS Code**) com terminal integrado.
* Variáveis de ambiente configuradas para reconhecer o utilitário `oplrun`.

## 📁 Estrutura de Arquivos
* `modelo.mod`: Contém a formulação matemática, conjuntos, variáveis de decisão e restrições em sintaxe OPL.
* `dados.dat`: Arquivo de dados isolado que carrega as matrizes de distância, demandas locais, parâmetros globais (como limite de caminhada e $droutemax$) e tuplas de configuração.

## 🚀 Como Executar (via VS Code)
1. Abra o terminal integrado no VS Code na pasta raiz do projeto.
2. Certifique-se de que os arquivos `.mod` e `.dat` estejam no mesmo diretório.
3. Execute o solver através do utilitário de linha de comando passando ambos os arquivos:
   ```bash
   oplrun modelo.mod dados.dat
O CPLEX processará as matrizes esparsas, otimizará o modelo e exibirá o log de resultados (incluindo as paradas selecionadas e os valores das funções objetivo) diretamente no terminal.
