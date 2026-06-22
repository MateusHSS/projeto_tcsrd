"""Script for training and fine-tuning a PPO GAT policy on the fault injection environment."""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from ambiente_mesh import AmbienteInjecaoFalhas, carregar_env
from extrator_gat import ExtratorFeaturesGAT

def main():
    """Main function to run the PPO GAT training workflow."""
    env_config = carregar_env()
    USAR_NS3 = env_config.get("USAR_NS3", "False").lower() in ("true", "1", "yes")
    NS3_PATH = env_config.get("NS3_PATH")
    
    if USAR_NS3:
        if not NS3_PATH or NS3_PATH == "/home/user/ns-3.48":
            raise ValueError("NS3_PATH must be explicitly configured in the environment or .env file when USAR_NS3=True.")
        if not os.path.exists(NS3_PATH):
            raise FileNotFoundError(f"The specified NS3_PATH does not exist: {NS3_PATH}")
    else:
        NS3_PATH = NS3_PATH or "/home/user/ns-3.48"

    NUM_NOS = 50
    TOTAL_PASSOS = 5000 if USAR_NS3 else 200000

    print(
        f"1. Instanciando os Ambientes Paralelos de Treinamento (Modo NS-3: {USAR_NS3})..."
    )
    num_envs = 2 if USAR_NS3 else 4
    caminho_instancias = "instances/train_50.csv"

    env = make_vec_env(
        lambda: AmbienteInjecaoFalhas(
            num_nos=NUM_NOS,
            usar_ns3=USAR_NS3,
            ns3_path=NS3_PATH,
            caminho_instancias=caminho_instancias,
        ),
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv,
    )

    print("2. Configurando a Arquitetura Híbrida (GAT + PPO)...")
    policy_kwargs = dict(
        features_extractor_class=ExtratorFeaturesGAT,
        features_extractor_kwargs=dict(features_dim=256, num_nos=NUM_NOS),
    )

    caminho_baseline = "modelos_pre_treinados/escala_50_ppo_gat.zip"

    if USAR_NS3 and os.path.exists(caminho_baseline):
        print(
            f"2. [Transfer Learning] Carregando cérebro GAT pré-treinado no NetworkX para calibrar no NS-3: {caminho_baseline}"
        )
        modelo_ppo_gat = PPO.load(
            caminho_baseline,
            env=env,
            learning_rate=0.0001,
            tensorboard_log="./escala_50_ppo_gat_ns3/",
        )
    else:
        modelo_ppo_gat = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=0.0003,
            n_steps=1024,
            ent_coef=0.01,
            tensorboard_log=(
                "./escala_50_ppo_gat_ns3/" if USAR_NS3 else "./escala_50_ppo_gat/"
            ),
            device="cpu",
        )

    print(
        f"3. Iniciando o Treinamento Baseado em Atenção Estrutural ({TOTAL_PASSOS} passos)..."
    )
    modelo_ppo_gat.learn(total_timesteps=TOTAL_PASSOS, progress_bar=True)

    print("4. Treinamento Concluído! Salvando o Framework Proposto...")

    nome_salvar = (
        "modelos_pre_treinados/escala_50_ppo_gat_ns3"
        if USAR_NS3
        else "modelos_pre_treinados/escala_50_ppo_gat"
    )
    modelo_ppo_gat.save(nome_salvar)

    print(f"[OK] Modelo definitivo salvo com sucesso em '{nome_salvar}.zip'")

if __name__ == "__main__":
    main()
