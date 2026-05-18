import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx


class AmbienteInjecaoFalhas(gym.Env):
    def __init__(self, num_nos=20):
        super(AmbienteInjecaoFalhas, self).__init__()
        self.num_nos = num_nos

        # 1. ESPAÇO DE AÇÃO: O PPO escolhe um número de 0 a (N-1) para atacar
        self.action_space = spaces.Discrete(self.num_nos)

        # 2. ESPAÇO DE OBSERVAÇÃO: A IA precisa ler o estado dos nós e quem está vivo
        # Para simplificar a integração com o PPO agora, vamos retornar um vetor achatado
        # contendo o status de cada nó (1 = Vivo, 0 = Destruído) e suas features.
        tamanho_observacao = self.num_nos * 3  # (Status, CPU, Memória) para cada nó
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(tamanho_observacao,), dtype=np.float32)

    # FUNÇÃO EXECUTADA A CADA NOVO EPISÓDIO PARA GERAR UMA REDE NOVA
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        semente_networkx = int(self.np_random.integers(0, 1000000))

        while True:
            self.G = nx.random_geometric_graph(self.num_nos, radius=0.35, seed=semente_networkx)
            if nx.is_connected(self.G):
                break
            semente_networkx = int(self.np_random.integers(0, 1000000))

        for node in self.G.nodes():
            self.G.nodes[node]['status'] = 1.0
            self.G.nodes[node]['features'] = [
                self.np_random.uniform(0.1, 0.9),
                self.np_random.uniform(0.2, 0.8)
            ]

        self.passos_dados = 0

        # NOVIDADE: Precisamos gravar o estado "saudável" da rede para comparar depois
        self.latencia_original = nx.average_shortest_path_length(self.G)
        self.redundancia_original = nx.average_node_connectivity(self.G)

        return self._obter_observacao(), {}

    def _obter_observacao(self):
        # Transforma o grafo do NetworkX em um vetor NumPy para o PPO conseguir ler
        obs = []
        for i in range(self.num_nos):
            status = self.G.nodes[i]['status']
            cpu, mem = self.G.nodes[i]['features']
            obs.extend([status, cpu, mem])
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.passos_dados += 1
        recompensa = 0.0
        terminou = False
        truncou = False

        if self.passos_dados >= 15:
            truncou = True

        if self.G.nodes[action]['status'] == 0.0:
            recompensa = -5.0
            return self._obter_observacao(), recompensa, terminou, truncou, {}

        self.G.nodes[action]['status'] = 0.0
        arestas_para_remover = list(self.G.edges(action))
        self.G.remove_edges_from(arestas_para_remover)

        nos_vivos = [n for n in self.G.nodes() if self.G.nodes[n]['status'] == 1.0]
        subgrafo = self.G.subgraph(nos_vivos)

        # --- O NOVO JUIZ MULTI-CRITÉRIO ---
        info = {}

        # 1. Checagem Crítica: A rede partiu? (Liveness)
        if not nx.is_connected(subgrafo):
            recompensa = +100.0
            terminou = True
            info['propriedade_violada'] = "Liveness (Conectividade Rompida - Rede Particionada)"
        else:
            # 2. Avaliação de Degradação (Safety & Multi-path)
            latencia_atual = nx.average_shortest_path_length(subgrafo)
            redundancia_atual = nx.average_node_connectivity(subgrafo)

            # Limiar de Violação de SLA de Tempo
            if latencia_atual >= (self.latencia_original * 1.5): # +50% da latência original
                recompensa = +100.0
                terminou = True
                info['propriedade_violada'] = f"Safety (Latência aumentou 50%+. Original: {self.latencia_original:.2f} | Atual: {latencia_atual:.2f})"
            else:
                # Recompensa Parcial
                aumento_latencia = latencia_atual - self.latencia_original
                perda_redundancia = self.redundancia_original - redundancia_atual

                recompensa = (aumento_latencia * 5.0) + (perda_redundancia * 2.0) - 1.0
                info['propriedade_violada'] = "Nenhuma (Ataque em andamento)"

        return self._obter_observacao(), recompensa, terminou, truncou, info