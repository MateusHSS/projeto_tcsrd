# Property-Guided Fault Injection in Mesh Networks Using Reinforcement Learning

This repository contains an Artificial Intelligence framework designed to evaluate the resilience of dynamic mesh networks through directed and automated attacks. The system uses a **Hybrid Deep Reinforcement Learning (DRL) architecture** with **Graph Attention Networks (GAT)** to discover complex topological vulnerabilities in graphs, assessing key communication properties beyond basic connectivity.

The framework serves as the decision engine for large-scale fault injection simulations in the **NS-3 discrete event simulator**, utilizing a cutting-edge **Transfer Learning** approach.

---

## 🧠 System Architecture

The framework is a sophisticated hybrid ecosystem combining Graph Neural Networks and Proximal Policy Optimization (PPO), divided into four core blocks:

### 1. State Representation (Observation Space)
The architecture captures the real-time network state in three matrices:
*   **Node Features:** Node status (alive/dead), Betweenness Centrality, and Degree.
*   **Edge Index:** The adjacency matrix dictating the physical layout.
*   **Performance Metrics:** Real-time Latency, Packet Delivery Ratio (PDR), and Delay.
*   **Action Masking:** A logical filter preventing the agent from attacking already offline routers, optimizing gradient flow and preventing infinite loops.

### 2. Neural Backbone (Custom GAT Extractor)
Built with `PyTorch Geometric`, this module replaces traditional flattened MLPs:
*   **Graph Attention Layers (GATConv):** Computes dynamic mathematical attention weights for neighboring nodes, allowing the AI to autonomously learn which routers are structural bottlenecks.
*   **Global Pooling:** Condenses the processed graph into a dense latent feature vector.

### 3. The Decision Brain (PPO Actor-Critic)
The dense vector is fed into the Proximal Policy Optimization (PPO) algorithm:
*   **Actor Network:** Outputs a probability distribution pointing to the exact node to be attacked.
*   **Critic Network:** Evaluates the expected reward of the chosen attack.

### 4. Dual-Engine Simulator
The environment dynamically routes the simulation to two distinct engines:
*   **Abstract Engine (NetworkX):** High-speed mathematical graph simulator. Used to force the AI to learn structural topology massively.
*   **Physical Engine (NS-3):** Deep C++ integration simulating Wi-Fi radio waves, OLSR/AODV routing protocols, Bit Error Rates (BER), and real packet collisions.

---

## 🚀 The Training Pipeline (Transfer Learning & Reward Shaping)

To train an AI capable of destroying physical NS-3 networks efficiently, we employ a 2-phase pipeline:

1. **Phase 1: Abstract Pre-Training (NetworkX)**
   - The agent plays 200,000 steps in mathematical graphs.
   - It learns the "theory" of latency and bottlenecks at ultra-high speeds.
2. **Phase 2: Physical Fine-Tuning (NS-3 Transfer Learning)**
   - The pre-trained brain is injected into the NS-3 physical simulator for 5,000 steps.
   - **The Sparse Reward Problem:** Since physical networks are highly resilient, finding the exact sequence to break the SLA (+100 reward) is statistically improbable, causing gradient collapse if strict penalties are used.
   - **The Solution (Reward Shaping):** We implemented a continuous reward mechanism that yields fractional points based on latency degradation, serving as "breadcrumbs" to guide the GAT surgically towards the physical bottleneck.

---

## 📁 Repository Structure

```text
├── run_all.sh               # Master script: Runs the entire 4-step pipeline automatically
├── resume_training.sh       # Recovery script: Resumes training from Phase 4 if interrupted
├── benchmark.py             # Benchmark execution script comparing models and random baseline
├── plot_benchmark.py        # Generates academic Bar charts, Boxplots, and Survival Curves
├── plot_learning_curve.py   # Generates continuous Transfer Learning evolution curves
├── mesh_environment.py      # Gymnasium environment class (supports NetworkX and NS-3)
├── gat_extractor.py         # Feature extractor using Graph Attention Networks (GAT)
├── instance_generator.py    # Parameterized script to generate fixed network topologies
├── mesh_simulation.cc       # Physical C++ simulation script for NS-3
├── train_ppo.py             # Training and fine-tuning script using MLP policies
├── train_ppo_gat.py         # Training and fine-tuning script using GAT policies
├── instances/               # Folder containing generated network topologies (CSV)
├── modelos_pre_treinados/   # Directory containing pre-trained model checkpoints
└── README.md                # System documentation
```

---

## 🛠️ How to Run

### 1. Installation
Set up a virtual environment (Python 3.8+) and install the dependencies:
```bash
pip install -r requirements.txt
```
*(Ensure `ns-3.48` is installed in your system and properly referenced in the `.env` file).*

### 2. The One-Click Pipeline
To execute the entire lifecycle (NetworkX Pre-training -> NS-3 Fine-tuning -> Benchmarking -> Graph Generation), simply run:
```bash
./run_all.sh
```
*(If your system reboots midway, you can resume by running `./resume_training.sh`).*

### 3. Generating the Academic Plots
After the benchmark completes, you can generate 5 high-resolution graphs ready for LaTeX/academic papers:
```bash
# Generates Bar charts, Boxplots of Stability, and Kaplan-Meier Survival Curves
python plot_benchmark.py

# Generates the Continuous Timeline comparing MLP and GAT learning speeds
python plot_learning_curve.py
```

---

## 📊 Validated Communication Properties

The framework calculates dense continuous rewards based on the following SLAs:

| Property | Mathematical Metric | Physical NS-3 Metric | Critical Violation |
| :--- | :--- | :--- | :--- |
| **Liveness** | `nx.is_connected(G)` | Packet Delivery Ratio (PDR) | Network partitions or PDR drops significantly. |
| **Safety** | `nx.average_shortest_path_length` | End-to-End Delay | Delay increases by 50%+ compared to baseline. |
| **Availability**| `nx.average_node_connectivity` | Throughput / Reachability | Extreme loss of structural redundancy. |
