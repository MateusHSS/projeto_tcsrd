# Property-Guided Fault Injection in Mesh Networks Using Reinforcement Learning

This repository contains an Artificial Intelligence framework designed to evaluate the resilience of dynamic mesh networks through directed and automated attacks. The system uses Deep Reinforcement Learning (PPO) to discover complex topological vulnerabilities in graphs, assessing key communication properties beyond basic connectivity.

The framework serves as the decision engine for large-scale fault injection simulations in the NS-3 discrete event simulator.

---

## System Architecture

The framework is structured using standard Python reinforcement learning libraries, using Gymnasium for environment modeling and Stable-Baselines3 for training algorithms.

1. Environment (AmbienteInjecaoFalhas): Models the Mesh network as a random geometric graph using NetworkX. Each node has dynamic attributes such as activation status and internal metrics.
2. Multi-Criteria Evaluator: For every agent action, the environment computes the real-time impact on the topology and penalizes or rewards the agent based on Service Level Agreements (SLA).
3. Agent (PPO): A neural network that observes the current state of the mesh and decides which critical node to disable to maximize system degradation.

---

## Validated Communication Properties

Unlike traditional approaches that only check if the network partitioned, this environment calculates continuous rewards (dense rewards) based on the following metrics:

| Property | Mathematical Metric (NetworkX) | Critical Violation Criterion |
| :--- | :--- | :--- |
| Liveness (Connectivity) | nx.is_connected(G) | The network partitions into two or more disconnected components. |
| Safety (End-to-End Latency) | nx.average_shortest_path_length(G) | The average shortest path length increases by 50% or more compared to the healthy network. |
| Multi-path Availability | nx.average_node_connectivity(G) | Measures residual redundancy (how many nodes need to fail to disconnect the network). |

---

## Repository Structure

```text
├── ambiente_mesh.py         # Gymnasium environment class (supports NetworkX and NS-3)
├── extrator_gat.py          # Feature extractor using Graph Attention Networks (GAT)
├── instance_generator.py    # Parameterized script to generate fixed network topologies
├── mesh_simulation.cc       # Physical C++ simulation script for NS-3
├── train_ppo.py             # Training and fine-tuning script using MLP policies
├── train_ppo_gat.py         # Training and fine-tuning script using GAT policies
├── validate_agent.py        # Validation script to load trained agents and run tests
├── instances/               # Folder containing generated network topologies (CSV)
│   ├── train_50.csv        # Fixed training topology instances (50 nodes)
│   └── val_20.csv           # Fixed validation topology instances (20 nodes)
├── modelos_pre_treinados/   # Directory containing pre-trained model checkpoints
│   ├── baseline_ppo_mlp.zip      # MLP policy trained on NetworkX
│   ├── escala_50_ppo_mlp.zip     # MLP policy for 50-node scale
│   └── escala_50_ppo_gat.zip     # GAT policy for 50-node scale
└── README.md                # System documentation
```

---

## Switching Between NetworkX and NS-3 (.env File)

To toggle between fast mathematical simulation using NetworkX and real packet-level traffic simulation using NS-3, edit the variables in the `.env` file in the repository root:

* USE_NS3: Set to True to enable NS-3 simulations, or False to use NetworkX.
* NS3_PATH: Path to your local NS-3 installation directory (e.g., /home/username/ns-3.48).

All training and validation scripts read this configuration dynamically.

---

## How to Run

1. Installation

Set up a virtual environment (Python 3.8+) and install the dependencies:

```bash
pip install -r requirements.txt
```

2. Generating Fixed Network Topologies

To generate reproducible datasets of network configurations, run the generator script:

```bash
# Generate 100 training instances with 50 nodes
python instance_generator.py --num_instances 100 --num_nodes 50 --output instances/train_50.csv --seed 42

# Generate 20 validation instances with 20 nodes
python instance_generator.py --num_instances 20 --num_nodes 20 --output instances/val_20.csv --seed 100
```

3. Training the Agent

To start training the policy using the generated training dataset:

```bash
python train_ppo.py
```

Or for GAT training:

```bash
python train_ppo_gat.py
```

The resulting model is automatically saved to the modelos_pre_treinados/ directory.

4. Validation

To run validation on a fixed network configuration from the generated validation dataset (using a specific instance ID like 0 for reproducibility):

```bash
python validate_agent.py
```

---

## Validation Output Example

When running the validation script, the step-by-step attack decisions are displayed:

```text
======================================================
 CARREGANDO A IA TREINADA (Modo NS-3: True)
======================================================
[Aviso] Modelo modelos_pre_treinados/escala_50_ppo_mlp_ns3 não encontrado. Tentando baseline_ppo_mlp...
[OK] Modelo carregado com sucesso!

======================================================
 INICIANDO O ATAQUE GUIADO PELA IA
======================================================
Passo 01 | IA atacou o Nó 08 | Recompensa: 100.00

======================================================
 RELATÓRIO FINAL DA INJEÇÃO DE FALHAS
======================================================
Status: [SUCESSO DO ATAQUE]
Total de Ataques Necessários : 1
Propriedade Violada          : Safety (Atraso médio aumentou 50%+. Original: 90.86ms | Atual: 236.05ms)
Recompensa Acumulada         : 100.00
======================================================
```
