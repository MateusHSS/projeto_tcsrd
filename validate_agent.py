"""Script de validação para carregar agentes pré-treinados e executar testes de injeção de falhas em instâncias de topologia fixa."""

import os
from stable_baselines3 import PPO
from mesh_environment import FaultInjectionEnvironment, load_env


def main():
    """Função principal para executar o fluxo de simulação e validação do agente."""
    env_config = load_env()
    USE_NS3 = env_config.get("USE_NS3", "False").lower() in ("true", "1", "yes")
    NS3_PATH = env_config.get("NS3_PATH")

    if USE_NS3:
        if not NS3_PATH or NS3_PATH == "/home/user/ns-3.48":
            raise ValueError(
                "NS3_PATH must be explicitly configured in the environment or .env file when USE_NS3=True."
            )
        if not os.path.exists(NS3_PATH):
            raise FileNotFoundError(
                f"The specified NS3_PATH does not exist: {NS3_PATH}"
            )
    else:
        NS3_PATH = NS3_PATH or "/home/user/ns-3.48"

    NUM_NODES = 20

    print("======================================================")
    print(f" CARREGANDO A IA TREINADA (Modo NS-3: {USE_NS3})")
    print("======================================================")

    instances_path = "instances/val_20.csv"
    env = FaultInjectionEnvironment(
        num_nodes=NUM_NODES,
        use_ns3=USE_NS3,
        ns3_path=NS3_PATH,
        instances_path=instances_path,
    )

    model_path = (
        "modelos_pre_treinados/escala_50_ppo_mlp_ns3"
        if USE_NS3
        else "modelos_pre_treinados/baseline_ppo_mlp"
    )

    try:
        model = PPO.load(model_path)
    except Exception:
        print(
            f"[Aviso] Modelo {model_path} não encontrado. Tentando baseline_ppo_mlp..."
        )
        model = PPO.load("modelos_pre_treinados/baseline_ppo_mlp.zip")

    print("[OK] Modelo carregado com sucesso!\n")

    print("======================================================")
    print(" INICIANDO O ATAQUE GUIADO PELA IA")
    print("======================================================")

    obs, _ = env.reset(options={"instancia_id": 0})
    episode_terminated = False
    step_count = 0
    total_reward = 0
    failure_reason = "A rede sobreviveu ao limite de ataques."

    while not episode_terminated:
        step_count += 1
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(
            f"Passo {step_count:02d} | IA atacou o Nó {action:02d} | Recompensa: {reward:6.2f}"
        )

        if terminated:
            failure_reason = info.get(
                "propriedade_violada", "Erro: Motivo não registrado."
            )

        episode_terminated = terminated or truncated

    print("\n======================================================")
    print(" RELATÓRIO FINAL DA INJEÇÃO DE FALHAS")
    print("======================================================")
    if total_reward > 0:
        print("Status: [SUCESSO DO ATAQUE]")
    else:
        print("Status: [FALHA DO ATAQUE - RESILIÊNCIA COMPROVADA]")

    print(f"Total de Ataques Necessários : {step_count}")
    print(f"Propriedade Violada          : {failure_reason}")
    print(f"Recompensa Acumulada         : {total_reward:.2f}")
    print("======================================================")


if __name__ == "__main__":
    main()

