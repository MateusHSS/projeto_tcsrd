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
├── ambiente_mesh.py         # Classe do ambiente Gymnasium (Regras do jogo e Recompensas)
├── treinamento_ppo.py       # Script para treino em larga escala (200.000 passos paralelos)
├── validacao_agente.py      # Script de teste para carregar o modelo e gerar relatórios
├── modelos_pre_treinados/   # Pasta binária para os melhores cérebros (ignorada em parte pelo Git)
│   └── baseline_inicial_200k.zip # Modelo treinado atual (Baseline usando rede linear)
└── README.md                # Esta documentação
```

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
