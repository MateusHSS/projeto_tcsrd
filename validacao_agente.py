"""Validation script to load pre-trained agents and run fault injection testing on fixed mesh topology instances."""

import os
from stable_baselines3 import PPO
from ambiente_mesh import AmbienteInjecaoFalhas, carregar_env

def main():
    """Main function to run the validation agent simulation workflow."""
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

    NUM_NOS = 20

    print("======================================================")
    print(f" CARREGANDO A IA TREINADA (Modo NS-3: {USAR_NS3})")
    print("======================================================")

    caminho_instancias = "instances/val_20.csv"
    env = AmbienteInjecaoFalhas(num_nos=NUM_NOS, usar_ns3=USAR_NS3, ns3_path=NS3_PATH, caminho_instancias=caminho_instancias)

    caminho_modelo = "modelos_pre_treinados/escala_50_ppo_mlp_ns3" if USAR_NS3 else "modelos_pre_treinados/baseline_ppo_mlp"

    try:
        modelo = PPO.load(caminho_modelo)
    except Exception:
        print(f"[Aviso] Modelo {caminho_modelo} não encontrado. Tentando baseline_ppo_mlp...")
        modelo = PPO.load("modelos_pre_treinados/baseline_ppo_mlp.zip")

    print("[OK] Modelo carregado com sucesso!\n")

    print("======================================================")
    print(" INICIANDO O ATAQUE GUIADO PELA IA")
    print("======================================================")

    obs, _ = env.reset(options={"instancia_id": 0})
    episodio_terminou = False
    passo = 0
    recompensa_total = 0
    motivo_da_queda = "A rede sobreviveu ao limite de ataques."

    while not episodio_terminou:
        passo += 1
        acao, _ = modelo.predict(obs, deterministic=True)
        acao = int(acao)

        obs, recompensa, terminou, truncou, info = env.step(acao)
        recompensa_total += recompensa

        print(f"Passo {passo:02d} | IA atacou o Nó {acao:02d} | Recompensa: {recompensa:6.2f}")

        if terminou:
            motivo_da_queda = info.get('propriedade_violada', 'Erro: Motivo não registrado.')

        episodio_terminou = terminou or truncou

    print("\n======================================================")
    print(" RELATÓRIO FINAL DA INJEÇÃO DE FALHAS")
    print("======================================================")
    if recompensa_total > 0:
        print("Status: [SUCESSO DO ATAQUE]")
    else:
        print("Status: [FALHA DO ATAQUE - RESILIÊNCIA COMPROVADA]")

    print(f"Total de Ataques Necessários : {passo}")
    print(f"Propriedade Violada          : {motivo_da_queda}")
    print(f"Recompensa Acumulada         : {recompensa_total:.2f}")
    print("======================================================")

if __name__ == "__main__":
    main()