from stable_baselines3 import PPO
from ambiente_mesh import AmbienteInjecaoFalhas

print("======================================================")
print(" CARREGANDO A INTELIGÊNCIA ARTIFICIAL TREINADA")
print("======================================================")

env = AmbienteInjecaoFalhas(num_nos=20)
modelo = PPO.load("modelos_pre_treinados/baseline_inicial_200k.zip")
print("[OK] Modelo carregado com sucesso!\n")

print("======================================================")
print(" INICIANDO O ATAQUE GUIADO PELA IA")
print("======================================================")

obs, _ = env.reset()
episodio_terminou = False
passo = 0
recompensa_total = 0
motivo_da_queda = "A rede sobreviveu ao limite de ataques."  # Mensagem padrão

while not episodio_terminou:
    passo += 1
    acao, _ = modelo.predict(obs, deterministic=True)
    acao = int(acao)

    obs, recompensa, terminou, truncou, info = env.step(acao)
    recompensa_total += recompensa

    print(f"Passo {passo:02d} | IA atacou o Nó {acao:02d} | Recompensa: {recompensa:6.2f}")

    # Se a rede quebrou neste turno, nós salvamos o motivo exato extraído do ambiente!
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