from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import os

# Importamos o nosso ambiente, o novo extrator e a função para ler o .env
from ambiente_mesh import AmbienteInjecaoFalhas, carregar_env
from extrator_gat import ExtratorFeaturesGAT

# --- CONFIGURAÇÃO DO SIMULADOR (CARREGADA AUTOMATICAMENTE DO .ENV) ---
env_config = carregar_env()
USAR_NS3 = env_config.get("USAR_NS3", "False").lower() in ("true", "1", "yes")
NS3_PATH = env_config.get("NS3_PATH", "/home/vinisilvag/ns-3.48")
NUM_NOS = 50  # Tamanho da rede Mesh a simular
TOTAL_PASSOS = 5000 if USAR_NS3 else 200000  # Menos passos se usar NS-3 (Fine-tuning)

if __name__ == '__main__':
    print(f"1. Instanciando os Ambientes Paralelos de Treinamento (Modo NS-3: {USAR_NS3})...")
    num_envs = 2 if USAR_NS3 else 4

    env = make_vec_env(
        lambda: AmbienteInjecaoFalhas(num_nos=NUM_NOS, usar_ns3=USAR_NS3, ns3_path=NS3_PATH),
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv
    )

    print("2. Configurando a Arquitetura Híbrida (GAT + PPO)...")
    policy_kwargs = dict(
        features_extractor_class=ExtratorFeaturesGAT,
        features_extractor_kwargs=dict(features_dim=256, num_nos=NUM_NOS),
    )

    caminho_baseline = "modelos_pre_treinados/escala_50_ppo_gat.zip"

    # Fine-tuning no NS-3 se o baseline do NetworkX já existir
    if USAR_NS3 and os.path.exists(caminho_baseline):
        print(f"2. [Transfer Learning] Carregando cérebro GAT pré-treinado no NetworkX para calibrar no NS-3: {caminho_baseline}")
        modelo_ppo_gat = PPO.load(
            caminho_baseline,
            env=env,
            learning_rate=0.0001,
            tensorboard_log="./escala_50_ppo_gat_ns3/"
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
            tensorboard_log=("./escala_50_ppo_gat_ns3/" if USAR_NS3 else "./escala_50_ppo_gat/"),
            device="cpu"
        )

    print(f"3. Iniciando o Treinamento Baseado em Atenção Estrutural ({TOTAL_PASSOS} passos)...")
    print("-> O processamento gráfico de mensagens em grafos exige mais computação.")
    modelo_ppo_gat.learn(total_timesteps=TOTAL_PASSOS, progress_bar=True)

    print("4. Treinamento Concluído! Salvando o Framework Proposto...")

    nome_salvar = "modelos_pre_treinados/escala_50_ppo_gat_ns3" if USAR_NS3 else "modelos_pre_treinados/escala_50_ppo_gat"
    modelo_ppo_gat.save(nome_salvar)

    print(f"[OK] Modelo definitivo salvo com sucesso em '{nome_salvar}.zip'")