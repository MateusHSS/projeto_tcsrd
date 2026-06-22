"""Script para treinamento e ajuste fino de uma política PPO GAT no ambiente de injeção de falhas."""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from mesh_environment import FaultInjectionEnvironment, load_env
from gat_extractor import GATFeaturesExtractor

def main():
    """Função principal para executar o fluxo de treinamento PPO GAT."""
    env_config = load_env()
    USE_NS3 = env_config.get("USE_NS3", "False").lower() in ("true", "1", "yes")
    NS3_PATH = env_config.get("NS3_PATH")
    
    if USE_NS3:
        if not NS3_PATH or NS3_PATH == "/home/user/ns-3.48":
            raise ValueError("NS3_PATH must be explicitly configured in the environment or .env file when USE_NS3=True.")
        if not os.path.exists(NS3_PATH):
            raise FileNotFoundError(f"The specified NS3_PATH does not exist: {NS3_PATH}")
    else:
        NS3_PATH = NS3_PATH or "/home/user/ns-3.48"

    NUM_NODES = 50
    TOTAL_STEPS = 5000 if USE_NS3 else 200000

    print(
        f"1. Instanciando os Ambientes Paralelos de Treinamento (Modo NS-3: {USE_NS3})..."
    )
    num_envs = 2 if USE_NS3 else 4
    instances_path = "instances/train_50.csv"

    env = make_vec_env(
        lambda: FaultInjectionEnvironment(
            num_nodes=NUM_NODES,
            use_ns3=USE_NS3,
            ns3_path=NS3_PATH,
            instances_path=instances_path,
        ),
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv,
    )

    print("2. Configurando a Arquitetura Híbrida (GAT + PPO)...")
    policy_kwargs = dict(
        features_extractor_class=GATFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256, num_nodes=NUM_NODES),
    )

    baseline_path = "modelos_pre_treinados/escala_50_ppo_gat.zip"

    if USE_NS3 and os.path.exists(baseline_path):
        print(
            f"2. [Transfer Learning] Carregando cérebro GAT pré-treinado no NetworkX para calibrar no NS-3: {baseline_path}"
        )
        ppo_gat_model = PPO.load(
            baseline_path,
            env=env,
            learning_rate=0.0001,
            tensorboard_log="./escala_50_ppo_gat_ns3/",
        )
    else:
        ppo_gat_model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=0.0003,
            n_steps=1024,
            ent_coef=0.01,
            tensorboard_log=(
                "./escala_50_ppo_gat_ns3/" if USE_NS3 else "./escala_50_ppo_gat/"
            ),
            device="cpu",
        )

    print(
        f"3. Iniciando o Treinamento Baseado em Atenção Estrutural ({TOTAL_STEPS} passos)..."
    )
    ppo_gat_model.learn(total_timesteps=TOTAL_STEPS, progress_bar=True)

    print("4. Treinamento Concluído! Salvando o Framework Proposto...")

    save_name = (
        "modelos_pre_treinados/escala_50_ppo_gat_ns3"
        if USE_NS3
        else "modelos_pre_treinados/escala_50_ppo_gat"
    )
    ppo_gat_model.save(save_name)

    print(f"[OK] Modelo definitivo salvo com sucesso em '{save_name}.zip'")

if __name__ == "__main__":
    main()
