# Framework para Injeção de Falhas Sensível à Topologia com Redes Neurais em Grafos e Aprendizado por Reforço

### 1. Instalação
```bash
pip install -r requirements.txt
```

### 2. Configurar o Ambiente (.env)
```text
USE_NS3=False
NS3_PATH=/home/user/ns-3.48
```

### 3. Gerando as Topologias
```bash
# Gerar instâncias de treinamento (50 nós)
python instance_generator.py --num_instances 100 --num_nodes 50 --output instances/train_50.csv --seed 42

# Gerar instâncias de benchmark (50 nós)
python instance_generator.py --num_instances 20 --num_nodes 50 --output instances/benchmark_50.csv --seed 100
```

### 4. Pré-Treinamento (NetworkX)
Coloque `USE_NS3=False` no seu `.env`

Rodar o treinamento da baseline MLP:
```bash
python train_ppo.py
```
Rodar o treinamento do framework GAT:
```bash
python train_ppo_gat.py
```

### 5. Transfer Learning
Coloque `USE_NS3=True` no seu `.env`

Rodar o fine-tuning da baseline MLP:
```bash
python train_ppo.py
```
Rodar o fine-tuning do framework GAT:
```bash
python train_ppo_gat.py
```

### 6. Executando o Benchmark
Coloque `USE_NS3=True` no `.env`:
```bash
python benchmark.py --num_nodes 50 --instances instances/benchmark_50.csv --use_ns3 True
```
