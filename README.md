# Injeção de Faltas Guiada por Propriedades em Redes Mesh usando Aprendizado por Reforço

Este repositório contém o framework de Inteligência Artificial desenvolvido para avaliar a resiliência de redes dinâmicas (Mesh/Roteamento) através de ataques direcionados e automatizados. O sistema utiliza **Aprendizado por Reforço Deep (PPO)** para descobrir vulnerabilidades topológicas complexas em grafos, avaliando propriedades fundamentais de comunicação além da conectividade básica.

O objetivo final deste framework é servir como o motor de decisão inteligente para simulações de injeção de faltas em larga escala no simulador de eventos discretos **NS-3**.

---

## 🏗️ Arquitetura do Sistema

O framework é estruturado seguindo o padrão do ecossistema de Aprendizado por Reforço em Python, utilizando `Gymnasium` para a modelação do ambiente e `Stable-Baselines3` para os algoritmos de treino.

1. **Ambiente (`AmbienteInjecaoFalhas`):** Modela a rede Mesh como um grafo geométrico aleatório utilizando `NetworkX`. Cada nó possui atributos dinâmicos (como estado de ativação e métricas internas).
2. **Juiz Multi-Critério:** A cada ação do agente, o ambiente calcula o impacto em tempo real na topologia e pune/premeia o agente baseado em acordos de nível de serviço (SLA).
3. **Agente (PPO):** Uma rede neuronal que observa o estado atual da malha e decide qual o nó crítico a derrubar para maximizar a degradação do sistema.

---

## 📊 Propriedades de Comunicação Validadas

Diferente de abordagens tradicionais que avaliam apenas se a rede se dividiu, este ambiente calcula penalizações contínuas (Dense Rewards) baseadas nas seguintes métricas da proposta de tese:


| Propriedade                      | Métrica Matemática (NetworkX)      | Critério de Violação Crítica                                                             |
| :------------------------------- | :----------------------------------- | :------------------------------------------------------------------------------------------- |
| **Liveness (Conectividade)**     | `nx.is_connected(G)`                 | A rede é particionada em dois ou mais componentes desconexos.                               |
| **Safety (Latência Fim-a-Fim)** | `nx.average_shortest_path_length(G)` | A latência média do caminho mínimo aumenta**50% ou mais** em relação à rede saudável. |
| **Multi-path Availability**      | `nx.average_node_connectivity(G)`    | Mede a redundância residual (quantos nós precisam de falhar para desconectar a rede).      |

---

## 📁 Estrutura do Repositório

```text
├── ambiente_mesh.py         # Classe do ambiente Gymnasium (Suporta NetworkX e NS-3)
├── mesh_simulation_sla.cc   # Simulação C++ física para o NS-3
├── treinamento_ppo.py       # Script para treino/fine-tuning (MlpPolicy)
├── treinamento_ppo_gat.py   # Script para treino/fine-tuning (GATPolicy)
├── validacao_agente.py      # Script de teste para carregar o modelo e gerar relatórios
├── modelos_pre_treinados/   # Pasta binária para os melhores cérebros
│   ├── baseline_ppo_mlp.zip      # Baseline treinado em NetworkX (MLP)
│   ├── escala_50_ppo_mlp.zip     # MLP de escala 50
│   └── escala_50_ppo_gat.zip     # GAT de escala 50
└── README.md                # Esta documentação
```

## ⚙️ Alternando entre NetworkX e NS-3 (Arquivo .env)

Para alternar entre a simulação matemática rápida do **NetworkX** e a simulação de tráfego física do **NS-3.48**, basta editar as variáveis no arquivo `.env` localizado na raiz do projeto:

* **`USAR_NS3`**: Defina como `True` para habilitar o NS-3 ou `False` para usar o NetworkX.
* **`NS3_PATH`**: Caminho da instalação do NS-3 (ex: `/home/vinisilvag/ns-3.48`).

Todos os scripts do repositório ([ambiente_mesh.py](file:///home/vinisilvag/ufmg/11%C2%BA%20Per%C3%ADodo/Confiabilidade%20em%20Sistemas%20de%20Redes%20Distribu%C3%ADdos/Projeto/projeto_tcsrd/ambiente_mesh.py), [treinamento_ppo.py](file:///home/vinisilvag/ufmg/11%C2%BA%20Per%C3%ADodo/Confiabilidade%20em%20Sistemas%20de%20Redes%20Distribu%C3%ADdos/Projeto/projeto_tcsrd/treinamento_ppo.py), [treinamento_ppo_gat.py](file:///home/vinisilvag/ufmg/11%C2%BA%20Per%C3%ADodo/Confiabilidade%20em%20Sistemas%20de%20Redes%20Distribu%C3%ADdos/Projeto/projeto_tcsrd/treinamento_ppo_gat.py) e [validacao_agente.py](file:///home/vinisilvag/ufmg/11%C2%BA%20Per%C3%ADodo/Confiabilidade%20em%20Sistemas%20de%20Redes%20Distribu%C3%ADdos/Projeto/projeto_tcsrd/validacao_agente.py)) lêem essa configuração dinamicamente a partir deste arquivo centralizado.

---

## 🚀 Como Executar

1. **Pré-requisitos**

Em um ambiente virtual com Python 3.8+, instale as dependências do projeto:


Bash

pip install -r requirements.txt

2. **Treinar o Agente do Zero**

Para iniciar o treino profundo com 4 ambientes paralelos e 200.000 passos de exploração (ajustado com coeficiente de entropia para maior criatividade do agente):
Bash

python treinamento_ppo.py

O modelo final será guardado automaticamente na pasta modelos_pre_treinados/baseline_ppo_mlp.zip.
3. **Executar a Validação / Teste de Injeção**

Para carregar o modelo treinado e ver a IA atacando a infraestrutura em tempo real, gerando o relatório final de propriedades violadas:
Bash

python validacao_agente.py

## 📉 Exemplo de Output da Validação

Quando o agente é executado, ele exibe passo a passo a sua estratégia de ataque. Repare nas recompensas intermediárias positivas, provando que a IA aprendeu a estrangular a latência antes de quebrar a conectividade:
Plaintext

======================================================
CARREGANDO A INTELIGÊNCIA ARTIFICIAL TREINADA
==============================================

[OK] Modelo carregado com sucesso da pasta de pré-treinados!

======================================================
INICIANDO O ATAQUE GUIADO PELA IA
=================================

Passo 01 | IA atacou o Nó 00 | Recompensa:   1.64
Passo 02 | IA atacou o Nó 13 | Recompensa:   1.67
Passo 03 | IA atacou o Nó 12 | Recompensa:   2.00
Passo 04 | IA atacou o Nó 08 | Recompensa: 100.00

======================================================
RELATÓRIO FINAL DA INJEÇÃO DE FALHAS
=======================================

Status: [SUCESSO DO ATAQUE]
Total de Ataques Necessários : 4
Propriedade Violada          : Liveness (Conectividade Rompida - Rede Particionada)
Recompensa Acumulada         : 105.31
