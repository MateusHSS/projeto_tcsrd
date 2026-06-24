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
├── benchmark.py             # Benchmark execution script comparing models and random baseline
├── mesh_environment.py      # Gymnasium environment class (supports NetworkX and NS-3)
├── gat_extractor.py         # Feature extractor using Graph Attention Networks (GAT)
├── instance_generator.py    # Parameterized script to generate fixed network topologies
├── mesh_simulation.cc       # Physical C++ simulation script for NS-3
├── train_ppo.py             # Training and fine-tuning script using MLP policies
├── train_ppo_gat.py         # Training and fine-tuning script using GAT policies
├── train_mlp_pipeline.sh    # Bash pipeline script for MLP training (NetworkX -> NS-3)
├── train_gat_pipeline.sh    # Bash pipeline script for GAT training (NetworkX -> NS-3)
├── validate_agent.py        # Validation script to load trained agents and run tests
├── instances/               # Folder containing generated network topologies (CSV)
│   ├── train_50.csv              # Fixed training topology instances (50 nodes)
│   └── benchmark_50.csv          # Benchmark topology instances (50 nodes)
├── benchmark_results.csv    # Detailed benchmark outputs
├── modelos_pre_treinados/   # Directory containing pre-trained model checkpoints
│   ├── escala_50_ppo_mlp.zip     # MLP policy for 50-node scale
│   └── escala_50_ppo_gat.zip     # GAT policy for 50-node scale
└── README.md                # System documentation
```

---

## Switching Between NetworkX and NS-3

The framework supports switching between fast mathematical graph simulation (NetworkX) and high-fidelity packet-level traffic simulation (NS-3). 

* **Simulator Path Configuration**: Configure `NS3_PATH` in the `.env` file at the repository root:
  ```text
  NS3_PATH=/home/username/ns-3.48
  ```
* **Training and Benchmarking Scripts**: The simulator choice is controlled directly via command-line flags/arguments (e.g., passing `--use_ns3`), allowing for seamless automation of training pipelines.
* **Validation Script**: `validate_agent.py` reads `USE_NS3` dynamically from the `.env` file to decide the simulation mode.

---

## Environment Customization Parameters

The `FaultInjectionEnvironment` supports additional configuration parameters in its constructor to mitigate training vulnerabilities:

* `include_topological_features` (default: False): Set to True to append Degree Centrality and Betweenness Centrality metrics to each node's state vector. This provides the MLP agent with explicit graph structure awareness.
* `penalize_milking` (default: False): Set to True to apply a constant step penalty of -2.0 for intermediate attacks instead of cumulative degradation rewards. This forces the RL agent to find the quickest path to network failure, preventing milking behaviors (reward hacking).

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

# Generate 20 benchmark/validation instances with 50 nodes
python instance_generator.py --num_instances 20 --num_nodes 50 --output instances/benchmark_50.csv --seed 100
```

3. Training the Agent

To start training using the fast mathematical simulation (NetworkX):

```bash
python train_ppo.py
python train_ppo_gat.py
```

To automate training using physical simulator (NS-3), pass the `--use_ns3` flag:

```bash
python train_ppo.py --use_ns3
python train_ppo_gat.py --use_ns3
```

Alternatively, you can run the automated pipeline scripts to execute the two stages sequentially (NetworkX training followed by NS-3 transfer learning/fine-tuning):

```bash
./train_mlp_pipeline.sh
./train_gat_pipeline.sh
```

The resulting model is automatically saved to the `modelos_pre_treinados/` directory.

4. Validation

To run validation on a fixed network configuration using the 50-node benchmark dataset (using a specific instance ID like 0 for reproducibility):

```bash
python validate_agent.py
```

5. Benchmarking

To run the benchmarking suite comparing the trained RL models against the random node-failure attack baseline:

```bash
# Run benchmark on 50-node instances using fast NetworkX simulation
python benchmark.py --num_nodes 50 --instances instances/benchmark_50.csv --use_ns3 False

# Run benchmark on 50-node instances using physical NS-3 simulation
python benchmark.py --num_nodes 50 --instances instances/benchmark_50.csv --use_ns3 True
```

The options available are:
* `--num_nodes`: Number of nodes in the network topology (default is 50).
* `--instances`: Path to the topology CSV file.
* `--use_ns3`: Override the USE_NS3 configuration from the `.env` file (accepts `True` or `False`).
* `--runs_random`: Number of runs for the random baseline to average results (default is 30).
* `--output`: Path to the output CSV file to write results (default is `./benchmark_results.csv`).
* `--seed`: Random seed for reproducibility (default is 42).

