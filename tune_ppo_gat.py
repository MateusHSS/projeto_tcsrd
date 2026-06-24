import optuna
import sys
import gat_extractor
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.modules['extrator_gat'] = gat_extractor

from mesh_environment import FaultInjectionEnvironment
from gat_extractor import ExtratorFeaturesGAT

def sample_ppo_params(trial):
    """Sorteia hiperparâmetros dentro de faixas seguras e comuns para o PPO."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "ent_coef": trial.suggest_float("ent_coef", 0.000001, 0.1, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048]),
        "gamma": trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99]),
        "clip_range": trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
    }

def objective(trial):
    hyperparams = sample_ppo_params(trial)
    
    # Treina sem o simulador NS-3 para ganhar velocidade e testar matematicamente as converções
    env = make_vec_env(
        lambda: FaultInjectionEnvironment(num_nodes=50, use_ns3=False, instances_path="instances/train_50.csv"),
        n_envs=4,
        vec_env_cls=SubprocVecEnv
    )
    
    policy_kwargs = dict(
        features_extractor_class=ExtratorFeaturesGAT,
        features_extractor_kwargs=dict(features_dim=256, num_nos=50),
    )
    
    model = PPO(
        "MlpPolicy", 
        env, 
        policy_kwargs=policy_kwargs, 
        device="cpu",
        **hyperparams
    )
    
    try:
        model.learn(total_timesteps=20000)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
    except Exception as e:
        env.close()
        raise optuna.exceptions.TrialPruned()
    
    env.close()
    return mean_reward

if __name__ == "__main__":
    print("Iniciando busca de hiperparâmetros (Optuna)...")
    study = optuna.create_study(direction="maximize")
    
    # Executa 30 iterações de teste
    study.optimize(objective, n_trials=30)
    
    print("Otimização Concluída!")
    print(f"Maior recompensa: {study.best_value}")

    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
