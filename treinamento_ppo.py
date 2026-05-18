from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from ambiente_mesh import AmbienteInjecaoFalhas

print("1. Instanciando os 4 Ambientes Paralelos...")
env = make_vec_env(lambda: AmbienteInjecaoFalhas(num_nos=20), n_envs=4)

print("2. Criando um NOVO Agente PPO (Resetando os Pesos)...")
modelo_ppo = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=1024,
    ent_coef=0.01,
    tensorboard_log="./baseline_inicial_200k/"
)

print("3. Iniciando o Treinamento Profundo (200.000 passos)...")
print("-> Dica: Vá pegar um café ou uma água, isso pode levar alguns minutos!")
modelo_ppo.learn(total_timesteps=200000, progress_bar=True)

print("4. Treinamento Concluído! Salvando o Novo Cérebro...")
modelo_ppo.save("modelos_pre_treinados/baseline_inicial_200k")

print("[OK] Agente salvo com sucesso em 'modelos_pre_treinados/baseline_inicial_200k.zip'")