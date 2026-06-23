"""Script de benchmarking para comparar o desempenho dos agentes de IA (MLP e GAT) contra a falha aleatória."""

import os
import sys
import time
import argparse
import random
import numpy as np
import pandas as pd

# Compatibilidade reversa para carregar modelos GAT salvos com nomes antigos de módulos/classes
try:
    import gat_extractor
    sys.modules['extrator_gat'] = gat_extractor
    gat_extractor.ExtratorFeaturesGAT = gat_extractor.GATFeaturesExtractor
    GAT_AVAILABLE = True
except ImportError:
    GAT_AVAILABLE = False

from stable_baselines3 import PPO
from mesh_environment import FaultInjectionEnvironment, load_env

def predict_action_any_size(model, obs, env):
    """Prediz a ação do modelo lidando com diferenças de tamanho de observação e ação.

    Adiciona preenchimento (padding) ou truncamento se o número de nós do modelo
    for diferente do número de nós do ambiente.
    """
    model_obs_len = model.observation_space.shape[0]
    env_obs_len = obs.shape[0]

    if env_obs_len == model_obs_len:
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    elif env_obs_len < model_obs_len:
        # O modelo espera mais nós do que o ambiente possui. Fazemos preenchimento (padding) com zeros.
        padded_obs = np.zeros(model_obs_len, dtype=np.float32)
        padded_obs[:env_obs_len] = obs
        action, _ = model.predict(padded_obs, deterministic=True)
        action = int(action)

        # Se a ação predita for um nó fictício (fora do limite do ambiente), escolhemos um nó vivo aleatório
        if action >= env.num_nodes:
            alive_nodes = [n for n in range(env.num_nodes) if env.G.nodes[n]['status'] == 1.0]
            if alive_nodes:
                action = int(np.random.choice(alive_nodes))
            else:
                action = 0
        return action

    else:
        # O ambiente tem mais nós do que o modelo suporta. Truncamos a observação.
        truncated_obs = obs[:model_obs_len]
        action, _ = model.predict(truncated_obs, deterministic=True)
        action = int(action)

        # Ajusta a ação caso ela ultrapasse o limite do ambiente
        if action >= env.num_nodes:
            action = action % env.num_nodes
        return action

def run_agent_episode(env, model, inst_id):
    """Executa um episódio completo para um agente baseado em modelo RL.

    Args:
        env (FaultInjectionEnvironment): O ambiente de simulação.
        model (PPO): O modelo PPO carregado.
        inst_id (int): O ID da instância a ser carregada.

    Returns:
        tuple: (número de passos, tempo decorrido, propriedade violada, sucesso)
    """
    start_time = time.time()
    obs, _ = env.reset(options={"instancia_id": inst_id})
    episode_terminated = False
    step_count = 0
    total_reward = 0
    failure_reason = "Survival"
    
    while not episode_terminated:
        step_count += 1
        action = predict_action_any_size(model, obs, env)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated:
            failure_reason = info.get('propriedade_violada', 'Violation')
            
        episode_terminated = terminated or truncated

    elapsed_time = time.time() - start_time
    success = total_reward > 0
    return step_count, elapsed_time, failure_reason, success

def run_random_episode(env, inst_id):
    """Executa um episódio completo utilizando uma política de escolha aleatória.

    Args:
        env (FaultInjectionEnvironment): O ambiente de simulação.
        inst_id (int): O ID da instância a ser carregada.

    Returns:
        tuple: (número de passos, tempo decorrido, propriedade violada, sucesso)
    """
    start_time = time.time()
    obs, _ = env.reset(options={"instancia_id": inst_id})
    episode_terminated = False
    step_count = 0
    total_reward = 0
    failure_reason = "Survival"
    
    while not episode_terminated:
        step_count += 1
        
        alive_nodes = [n for n in range(env.num_nodes) if env.G.nodes[n]['status'] == 1.0]
        if not alive_nodes:
            break
            
        action = int(np.random.choice(alive_nodes))
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated:
            failure_reason = info.get('propriedade_violada', 'Violation')
            
        episode_terminated = terminated or truncated

    elapsed_time = time.time() - start_time
    success = total_reward > 0
    return step_count, elapsed_time, failure_reason, success

def main():
    """Fluxo principal do benchmark."""
    parser = argparse.ArgumentParser(description='Benchmark of Fault Injection Strategies.')
    parser.add_argument('--num_nodes', type=int, default=50, help='Number of nodes (must match instance file).')
    parser.add_argument('--instances', type=str, default='instances/benchmark_50.csv', help='Path to the instances CSV file.')
    parser.add_argument('--use_ns3', type=str, default=None, choices=['True', 'False'], help='Override USE_NS3 configuration from .env.')
    parser.add_argument('--runs_random', type=int, default=5, help='Number of runs for the random baseline to average results.')
    parser.add_argument('--output', type=str, default='instances/benchmark_results.csv', help='Output CSV with results.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    args = parser.parse_args()

    # Define a semente para fins de reprodutibilidade
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    env_config = load_env()
    if args.use_ns3 is not None:
        USE_NS3 = args.use_ns3 == 'True'
    else:
        USE_NS3 = env_config.get("USE_NS3", "False").lower() in ("true", "1", "yes")
    NS3_PATH = env_config.get("NS3_PATH")
    
    if USE_NS3:
        if not NS3_PATH or NS3_PATH == "/home/user/ns-3.48":
            raise ValueError("NS3_PATH must be explicitly configured in the environment or .env file when USE_NS3=True.")
        if not os.path.exists(NS3_PATH):
            raise FileNotFoundError(f"The specified NS3_PATH does not exist: {NS3_PATH}")
    else:
        NS3_PATH = NS3_PATH or "/home/user/ns-3.48"

    print("======================================================")
    print(f" INICIANDO BENCHMARK DE FALHAS (Modo NS-3: {USE_NS3})")
    print("======================================================")
    print(f"Rede: {args.num_nodes} nós | Instâncias: {args.instances} | Seed: {args.seed}")

    env = FaultInjectionEnvironment(
        num_nodes=args.num_nodes,
        use_ns3=USE_NS3,
        ns3_path=NS3_PATH,
        instances_path=args.instances
    )

    if not env.instances:
        raise ValueError("O arquivo de instâncias está vazio ou não pôde ser carregado.")
    instance_ids = sorted(list(env.instances.keys()))
    print(f"Total de instâncias encontradas: {len(instance_ids)}")

    # Carrega o modelo MLP de forma resiliente
    mlp_model = None
    mlp_model_path = "modelos_pre_treinados/escala_50_ppo_mlp_ns3" if USE_NS3 else "modelos_pre_treinados/escala_50_ppo_mlp"
    try:
        mlp_model = PPO.load(mlp_model_path)
        print(f"[OK] Modelo MLP carregado com sucesso a partir de '{mlp_model_path}'.")
    except Exception as e_mlp_scale:
        print(f"[Aviso] Modelo MLP de escala 50 não pôde ser carregado de '{mlp_model_path}': {e_mlp_scale}")
        print("Tentando carregar modelo baseline_ppo_mlp...")
        try:
            mlp_model = PPO.load("modelos_pre_treinados/baseline_ppo_mlp.zip")
            print("[OK] Modelo MLP baseline (20 nós) carregado com sucesso.")
        except Exception as e_mlp_base:
            print(f"[Erro] Modelo MLP baseline não pôde ser carregado: {e_mlp_base}")

    # Carrega o modelo GAT de forma resiliente
    gat_model = None
    gat_model_path = "modelos_pre_treinados/escala_50_ppo_gat_ns3" if USE_NS3 else "modelos_pre_treinados/escala_50_ppo_gat"
    if GAT_AVAILABLE:
        try:
            gat_model = PPO.load(gat_model_path)
            print(f"[OK] Modelo GAT carregado com sucesso a partir de '{gat_model_path}'.")
        except Exception as e:
            print(f"[Aviso] Modelo GAT não pôde ser carregado de '{gat_model_path}': {e}")
    else:
        print("[Aviso] GAT desabilitado devido a dependências em falta (PyTorch Geometric).")

    results = []

    for inst_id in instance_ids:
        print(f"\nAvaliando Instância {inst_id}...")
        
        mlp_steps, mlp_time, mlp_reason, mlp_success = np.nan, np.nan, "N/A", False
        if mlp_model is not None:
            try:
                mlp_steps, mlp_time, mlp_reason, mlp_success = run_agent_episode(env, mlp_model, inst_id)
                print(f"  MLP Agent  -> Passos: {mlp_steps:02d} | Tempo: {mlp_time:6.3f}s | Status: {mlp_reason}")
            except Exception as e:
                print(f"  MLP Agent  -> Erro na execução: {e}")
        
        gat_steps, gat_time, gat_reason, gat_success = np.nan, np.nan, "N/A", False
        if gat_model is not None:
            try:
                gat_steps, gat_time, gat_reason, gat_success = run_agent_episode(env, gat_model, inst_id)
                print(f"  GAT Agent  -> Passos: {gat_steps:02d} | Tempo: {gat_time:6.3f}s | Status: {gat_reason}")
            except Exception as e:
                print(f"  GAT Agent  -> Erro na execução: {e}")
        
        rand_steps_list = []
        rand_time_list = []
        rand_reasons = []
        
        for _ in range(args.runs_random):
            r_steps, r_time, r_reason, _ = run_random_episode(env, inst_id)
            rand_steps_list.append(r_steps)
            rand_time_list.append(r_time)
            rand_reasons.append(r_reason)
            
        avg_rand_steps = float(np.mean(rand_steps_list))
        avg_rand_time = float(np.mean(rand_time_list))
        most_common_reason = max(set(rand_reasons), key=rand_reasons.count)
        print(f"  Random (x{args.runs_random}) -> Passos (Média): {avg_rand_steps:5.1f} | Tempo: {avg_rand_time:6.3f}s | Status: {most_common_reason}")

        results.append({
            "instance_id": inst_id,
            "mlp_steps": mlp_steps,
            "mlp_time_seconds": mlp_time,
            "mlp_status": mlp_reason,
            "gat_steps": gat_steps,
            "gat_time_seconds": gat_time,
            "gat_status": gat_reason,
            "random_steps_avg": avg_rand_steps,
            "random_time_seconds_avg": avg_rand_time,
            "random_status": most_common_reason
        })

    # Cria o DataFrame com os resultados usando pandas
    df = pd.DataFrame(results)
    
    # Salva os resultados detalhados em um arquivo CSV usando pandas
    df.to_csv(args.output, index=False)

    # Computa as estatísticas agregadas usando pandas
    mlp_steps_mean = df["mlp_steps"].mean()
    mlp_time_mean = df["mlp_time_seconds"].mean()
    
    gat_steps_mean = df["gat_steps"].mean()
    gat_time_mean = df["gat_time_seconds"].mean()
    
    rand_steps_mean = df["random_steps_avg"].mean()
    rand_time_mean = df["random_time_seconds_avg"].mean()

    print("\n======================================================")
    print(" RELATÓRIO FINAL DO BENCHMARK (MÉDIAS)")
    print("======================================================")
    if mlp_model is not None:
        print(f"Média Passos - MLP    : {mlp_steps_mean:.2f}")
    if gat_model is not None:
        print(f"Média Passos - GAT    : {gat_steps_mean:.2f}")
    print(f"Média Passos - Random : {rand_steps_mean:.2f}")
    print("------------------------------------------------------")
    if mlp_model is not None:
        print(f"Média Tempo  - MLP    : {mlp_time_mean:.3f}s")
    if gat_model is not None:
        print(f"Média Tempo  - GAT    : {gat_time_mean:.3f}s")
    print(f"Média Tempo  - Random : {rand_time_mean:.3f}s")
    print("======================================================")
    print(f"Resultados detalhados salvos em '{args.output}'")

if __name__ == "__main__":
    main()
