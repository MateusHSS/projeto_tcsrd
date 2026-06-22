"""Script for training and fine-tuning a PPO MLP policy on the fault injection environment."""

import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from ambiente_mesh import AmbienteInjecaoFalhas, carregar_env

def main():
    """Main function to run the PPO MLP training workflow."""
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

    print(f"1. Instanciando os Ambientes Paralelos (Modo NS-3: {USAR_NS3})...")

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

    caminho_baseline = "modelos_pre_treinados/escala_50_ppo_mlp.zip"

    if USAR_NS3 and os.path.exists(caminho_baseline):
        print(
            f"2. [Transfer Learning] Carregando cérebro pré-treinado no NetworkX para calibrar no NS-3: {caminho_baseline}"
        )
        modelo_ppo = PPO.load(
            caminho_baseline,
            env=env,
            learning_rate=0.0001,
            tensorboard_log="./escala_50_ppo_mlp_ns3/",
        )
    else:
        print("2. Criando o Agente PPO do zero (Baseline)...")
        modelo_ppo = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=1024,
            ent_coef=0.01,
            tensorboard_log=(
                "./escala_50_ppo_mlp_ns3/" if USAR_NS3 else "./escala_50_ppo_mlp/"
            ),
            device="cpu",
        )

    print(f"3. Iniciando o Treinamento Profundo ({TOTAL_PASSOS} passos)...")
    modelo_ppo.learn(total_timesteps=TOTAL_PASSOS, progress_bar=True)

    print("4. Treinamento Concluído!")

    nome_salvar = (
        "modelos_pre_treinados/escala_50_ppo_mlp_ns3"
        if USAR_NS3
        else "modelos_pre_treinados/escala_50_ppo_mlp"
    )
    modelo_ppo.save(nome_salvar)
    print(f"[OK] Modelo salvo com sucesso em '{nome_salvar}.zip'")

if __name__ == "__main__":
    main()
