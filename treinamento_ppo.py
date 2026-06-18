import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv  # <--- O Módulo de Paralelismo Real
from ambiente_mesh import AmbienteInjecaoFalhas, carregar_env
import os

# --- CONFIGURAÇÃO DO SIMULADOR (CARREGADA AUTOMATICAMENTE DO .ENV) ---
env_config = carregar_env()
USAR_NS3 = env_config.get("USAR_NS3", "False").lower() in ("true", "1", "yes")
NS3_PATH = env_config.get("NS3_PATH", "/home/vinisilvag/ns-3.48")
NUM_NOS = 50  # Tamanho da rede Mesh a simular
TOTAL_PASSOS = 5000 if USAR_NS3 else 200000  # Menos passos se usar NS-3 (Fine-tuning)

if __name__ == '__main__':
    print(f"1. Instanciando os Ambientes Paralelos (Modo NS-3: {USAR_NS3})...")

    # Limitamos a 2 ambientes paralelos em modo NS-3 para evitar sobrecarga de compilação simultânea
    num_envs = 2 if USAR_NS3 else 4

    env = make_vec_env(
        lambda: AmbienteInjecaoFalhas(num_nos=NUM_NOS, usar_ns3=USAR_NS3, ns3_path=NS3_PATH),
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv
    )

    caminho_modelo = "modelos_pre_treinados/escala_50_ppo_mlp"
    caminho_baseline = "modelos_pre_treinados/escala_50_ppo_mlp.zip"

    # Se estivermos rodando no NS-3 e um baseline NetworkX já existir, fazemos transferência de aprendizado (Fine-Tuning)
    if USAR_NS3 and os.path.exists(caminho_baseline):
        print(f"2. [Transfer Learning] Carregando cérebro pré-treinado no NetworkX para calibrar no NS-3: {caminho_baseline}")
        modelo_ppo = PPO.load(
            caminho_baseline,
            env=env,
            learning_rate=0.0001,  # Taxa menor para calibração fina
            tensorboard_log="./escala_50_ppo_mlp_ns3/"
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
            tensorboard_log=("./escala_50_ppo_mlp_ns3/" if USAR_NS3 else "./escala_50_ppo_mlp/"),
            device="cpu"
        )

    print(f"3. Iniciando o Treinamento Profundo ({TOTAL_PASSOS} passos)...")
    modelo_ppo.learn(total_timesteps=TOTAL_PASSOS, progress_bar=True)

    print("4. Treinamento Concluído!")
    
    nome_salvar = "modelos_pre_treinados/escala_50_ppo_mlp_ns3" if USAR_NS3 else "modelos_pre_treinados/escala_50_ppo_mlp"
    modelo_ppo.save(nome_salvar)
    print(f"[OK] Modelo salvo com sucesso em '{nome_salvar}.zip'")