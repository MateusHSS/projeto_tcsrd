from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# Importamos o nosso ambiente e o novo extrator que criamos
from ambiente_mesh import AmbienteInjecaoFalhas
from extrator_gat import ExtratorFeaturesGAT

if __name__ == '__main__':
    print("1. Instanciando os Ambientes Paralelos de Treinamento...")
    # Usaremos 4 ambientes em paralelo para acelerar o processo
    env = make_vec_env(
        lambda: AmbienteInjecaoFalhas(num_nos=50),
        n_envs=4,
        vec_env_cls=SubprocVecEnv
    )

    print("2. Configurando a Arquitetura Híbrida (GAT + PPO)...")
    policy_kwargs = dict(
        features_extractor_class=ExtratorFeaturesGAT,
        features_extractor_kwargs=dict(features_dim=256, num_nos=50),
    )

    modelo_ppo_gat = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=0.0003,
        n_steps=1024,
        ent_coef=0.01,
        tensorboard_log="./escala_50_ppo_gat/",
        device="cpu"
    )

    print("3. Iniciando o Treinamento Baseado em Atenção Estrutural (200.000 passos)...")
    print("-> O processamento gráfico de mensagens em grafos exige mais computação.")
    modelo_ppo_gat.learn(total_timesteps=200000, progress_bar=True)

    print("4. Treinamento Concluído! Salvando o Framework Proposto...")

    modelo_ppo_gat.save("modelos_pre_treinados/escala_50_ppo_gat")

    print("[OK] Modelo definitivo salvo com sucesso em 'modelos_pre_treinados/escala_50_ppo_gat.zip'")