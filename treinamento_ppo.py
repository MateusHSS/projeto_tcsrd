import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv  # <--- O Módulo de Paralelismo Real
from ambiente_mesh import AmbienteInjecaoFalhas

if __name__ == '__main__':
    print("1. Instanciando os 4 Ambientes Paralelos (Multiprocessamento REAL de CPU)...")

    env = make_vec_env(
        lambda: AmbienteInjecaoFalhas(num_nos=50),
        n_envs=4,
        vec_env_cls=SubprocVecEnv
    )

    print("2. Criando o Agente PPO (Baseline)...")
    modelo_ppo = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=1024,
        ent_coef=0.01,
        tensorboard_log="./escala_50_ppo_mlp/",
        device="cpu"
    )

    print("3. Iniciando o Treinamento Profundo (200.000 passos)...")
    modelo_ppo.learn(total_timesteps=200000, progress_bar=True)

    print("4. Treinamento Concluído!")
    modelo_ppo.save("modelos_pre_treinados/escala_50_ppo_mlp")