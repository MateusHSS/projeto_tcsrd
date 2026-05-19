import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx


class AmbienteInjecaoFalhas(gym.Env):
    def __init__(self, num_nos=20):
        super(AmbienteInjecaoFalhas, self).__init__()
        self.num_nos = num_nos

        if self.num_nos <= 20:
            self.raio_comunicacao = 0.35
            self.limite_passos = 15  # 75% da rede
        elif self.num_nos <= 50:
            self.raio_comunicacao = 0.22  # Evita que 50 nós formem um bloco indestrutível
            self.limite_passos = 35  # 70% da rede
        else:
            self.raio_comunicacao = 0.15
            self.limite_passos = int(self.num_nos * 0.7)

        # --- 2. ESPAÇO DE AÇÃO (A "Mão" do Agente) ---
        # A IA pode escolher um botão de 0 até (num_nos - 1)
        self.action_space = gym.spaces.Discrete(self.num_nos)

        # --- 3. ESPAÇO DE OBSERVAÇÃO (Os "Olhos" do Agente) ---
        # Matriz achatada: [Status, CPU, Memória] para cada nó.
        # Se num_nos = 50, a IA lerá um vetor dinâmico de 150 posições.
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_nos * 3,),
            dtype=np.float32
        )

        self.grafo = None
        self.passos_dados = 0

    # FUNÇÃO EXECUTADA A CADA NOVO EPISÓDIO PARA GERAR UMA REDE NOVA
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        semente_networkx = int(self.np_random.integers(0, 1000000))

        while True:
            self.G = nx.random_geometric_graph(self.num_nos, radius=self.raio_comunicacao)

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

        # --- OTIMIZAÇÃO MATEMÁTICA NO RESET ---
        self.latencia_original = nx.average_shortest_path_length(self.G)

        if self.num_nos <= 20:
            self.redundancia_original = nx.average_node_connectivity(self.G)
        else:
            # Usa a métrica global (muito mais leve e eficiente) para redes grandes
            self.redundancia_original = nx.node_connectivity(self.G)

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

        if self.passos_dados >= self.limite_passos:
            truncou = True

        if self.G.nodes[action]['status'] == 0.0:
            recompensa = -5.0
            return self._obter_observacao(), recompensa, terminou, truncou, {}

        self.G.nodes[action]['status'] = 0.0
        arestas_para_remover = list(self.G.edges(action))
        self.G.remove_edges_from(arestas_para_remover)

        nos_vivos = [n for n in self.G.nodes() if self.G.nodes[n]['status'] == 1.0]
        subgrafo = self.G.subgraph(nos_vivos)

        info = {}

        # 1. Checagem Crítica: A rede partiu? (Liveness)
        if not nx.is_connected(subgrafo):
            recompensa = +100.0
            terminou = True
            info['propriedade_violada'] = "Liveness (Conectividade Rompida - Rede Particionada)"
        else:
            # 2. Avaliação de Degradação (Safety & Multi-path)
            latencia_atual = nx.average_shortest_path_length(subgrafo)

            if self.num_nos <= 20:
                redundancia_atual = nx.average_node_connectivity(subgrafo)
            else:
                # Usa a métrica global para evitar explosão combinatória O(n^4)
                redundancia_atual = nx.node_connectivity(subgrafo)

            # Limiar de Violação de SLA de Tempo
            if latencia_atual >= (self.latencia_original * 1.5):  # +50% da latência original
                recompensa = +100.0
                terminou = True
                info[
                    'propriedade_violada'] = f"Safety (Latência aumentou 50%+. Original: {self.latencia_original:.2f} | Atual: {latencia_atual:.2f})"
            else:
                # Recompensa Parcial
                aumento_latencia = latencia_atual - self.latencia_original
                perda_redundancia = self.redundancia_original - redundancia_atual

                recompensa = (aumento_latencia * 5.0) + (perda_redundancia * 2.0) - 1.0
                info['propriedade_violada'] = "Nenhuma (Ataque em andamento)"

        return self._obter_observacao(), recompensa, terminou, truncou, info