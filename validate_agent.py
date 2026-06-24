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

    NUM_NODES = 50

    print(f"Loading trained model (NS-3: {USE_NS3})...")

    instances_path = "instances/benchmark_50.csv"
    env = FaultInjectionEnvironment(
        num_nodes=NUM_NODES,
        use_ns3=USE_NS3,
        ns3_path=NS3_PATH,
        instances_path=instances_path,
    )

    model_path = (
        "modelos_pre_treinados/escala_50_ppo_mlp_ns3"
        if USE_NS3
        else "modelos_pre_treinados/escala_50_ppo_mlp"
    )

    try:
        model = PPO.load(model_path, device="cpu")
    except Exception as e:
        raise FileNotFoundError(
            f"Could not load PPO model from '{model_path}': {e}"
        )

    print("[OK] Model loaded successfully.\n")

    print("Starting guided attacks...")

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
            f"Step {step_count:02d} | Node {action:02d} attacked | Reward: {reward:6.2f}"
        )

        if terminated:
            failure_reason = info.get(
                "propriedade_violada", "Erro: Motivo não registrado."
            )

        episode_terminated = terminated or truncated

    print("\n--- Final Fault Injection Report ---")
    if total_reward > 0:
        print("Status: [ATTACK SUCCESS]")
    else:
        print("Status: [ATTACK FAILED - RESILIENT NETWORK]")

    print(f"Total steps taken  : {step_count}")
    print(f"Property violated  : {failure_reason}")
    print(f"Cumulative reward  : {total_reward:.2f}")


if __name__ == "__main__":
    main()

