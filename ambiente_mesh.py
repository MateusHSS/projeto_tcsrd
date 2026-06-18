import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import os
import subprocess
import json
import shutil


def carregar_env():
    """Lê variáveis do arquivo .env de forma leve e sem dependências externas"""
    config = {"USAR_NS3": "False", "NS3_PATH": "/home/vinisilvag/ns-3.48"}
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_env = os.path.join(diretorio_atual, ".env")
    if os.path.exists(caminho_env):
        with open(caminho_env, "r") as f:
            for linha in f:
                linha = linha.strip()
                if linha and not linha.startswith("#") and "=" in linha:
                    chave, valor = linha.split("=", 1)
                    config[chave.strip()] = valor.strip()
    return config


class AmbienteInjecaoFalhas(gym.Env):
    def __init__(self, num_nos=20, usar_ns3=None, ns3_path=None):
        super(AmbienteInjecaoFalhas, self).__init__()
        self.num_nos = num_nos

        # Carrega configurações do .env se não passadas explicitamente
        env_config = carregar_env()
        
        if usar_ns3 is None:
            self.usar_ns3 = env_config.get("USAR_NS3", "False").lower() in ("true", "1", "yes")
        else:
            self.usar_ns3 = usar_ns3

        if ns3_path is None:
            self.ns3_path = env_config.get("NS3_PATH", "/home/vinisilvag/ns-3.48")
        else:
            self.ns3_path = ns3_path

        if self.num_nos <= 20:
            self.raio_comunicacao = 0.35
            self.limite_passos = 15  # 75% da rede
        elif self.num_nos <= 50:
            self.raio_comunicacao = 0.22  # Evita que 50 nós formem um bloco indestrutível
            self.limite_passos = 35  # 70% da rede
        else:
            self.raio_comunicacao = 0.15
            self.limite_passos = int(self.num_nos * 0.7)

        # Copia automaticamente o script de simulação C++ se o modo NS-3 estiver habilitado
        if self.usar_ns3:
            self._copiar_script_simulacao()

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
        self.failed_nodes = set()

    def _copiar_script_simulacao(self):
        """Garante que o arquivo C++ de simulação esteja na pasta scratch do NS-3"""
        diretorio_atual = os.path.dirname(os.path.abspath(__file__))
        src_cc = os.path.join(diretorio_atual, "mesh_simulation_sla.cc")
        dest_cc = os.path.join(self.ns3_path, "scratch", "mesh_simulation_sla.cc")

        if os.path.exists(src_cc):
            if not os.path.exists(dest_cc) or os.path.getmtime(src_cc) > os.path.getmtime(dest_cc):
                print(f"[NS3-Bridge] Copiando {src_cc} para {dest_cc}...")
                os.makedirs(os.path.dirname(dest_cc), exist_ok=True)
                shutil.copy(src_cc, dest_cc)
        else:
            print(f"[NS3-Bridge] Aviso: {src_cc} não encontrado localmente no projeto.")

    def _executar_simulacao_ns3(self):
        """Chama a simulação física C++ no NS-3 e captura os resultados estruturados em JSON"""
        # Passa "none" se o conjunto de nós falhos estiver vazio, evitando erro no parser do CommandLine do NS-3
        failed_nodes_str = ",".join(map(str, self.failed_nodes)) if self.failed_nodes else "none"
        
        # Coleta os valores de CPU e Memória de cada nó
        cpus = [self.G.nodes[i]['features'][0] for i in range(self.num_nos)]
        mems = [self.G.nodes[i]['features'][1] for i in range(self.num_nos)]
        
        cpu_str = ",".join(f"{c:.4f}" for c in cpus)
        mem_str = ",".join(f"{m:.4f}" for m in mems)

        # Extrai posições 2D do NetworkX e escala por 500
        pos = nx.get_node_attributes(self.G, 'pos')
        posicoes_lista = []
        for i in range(self.num_nos):
            x, y = pos[i]
            posicoes_lista.append(f"{x * 500.0:.2f},{y * 500.0:.2f}")
        pos_str = ",".join(posicoes_lista)

        # Range geométrico correspondente em metros
        range_fisico = self.raio_comunicacao * 500.0

        # Comando para executar no NS-3.48
        comando = [
            "./ns3", "run", "scratch/mesh_simulation_sla",
            "--",
            f"--numNodes={self.num_nos}",
            f"--failedNodes={failed_nodes_str}",
            f"--cpuValues={cpu_str}",
            f"--memValues={mem_str}",
            f"--nodePositions={pos_str}",
            f"--range={range_fisico}"
        ]

        try:
            resultado = subprocess.run(
                comando,
                cwd=self.ns3_path,
                capture_output=True,
                text=True,
                check=True
            )

            linhas = resultado.stdout.strip().split("\n")
            json_str = ""
            for linha in linhas:
                if linha.strip().startswith("{") and linha.strip().endswith("}"):
                    json_str = linha.strip()
                    break

            if not json_str:
                raise ValueError(f"NS-3 não retornou JSON válido. Stdout:\n{resultado.stdout}")

            return json.loads(json_str)

        except (subprocess.CalledProcessError, ValueError) as e:
            print(f"[NS3-Bridge] Falha na execução do simulador: {e}")
            return {"pdr": 0.0, "avg_delay": 9.99, "tx_packets": 0, "rx_packets": 0}

    # FUNÇÃO EXECUTADA A CADA NOVO EPISÓDIO PARA GERAR UMA REDE NOVA
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.failed_nodes.clear()
        self.passos_dados = 0
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

        if self.usar_ns3:
            # Medição no NS-3 saudável para linha de base
            dados_saudaveis = self._executar_simulacao_ns3()
            self.pdr_original = dados_saudaveis.get("pdr", 100.0)
            self.delay_original = dados_saudaveis.get("avg_delay", 0.001)
            if self.delay_original == 0.0:
                self.delay_original = 0.001
        else:
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

        if self.usar_ns3:
            self.failed_nodes.add(action)
            dados = self._executar_simulacao_ns3()
            pdr_atual = dados["pdr"]
            delay_atual = dados["avg_delay"]
            rx_packets = dados.get("rx_packets", 0)

            if rx_packets == 0 or pdr_atual == 0.0:
                delay_atual = 9.99

            info = {}

            # 1. Checagem Crítica: Liveness (PDR cai abaixo de 10% do baseline saudável)
            if pdr_atual <= (self.pdr_original * 0.1):
                recompensa = +100.0
                terminou = True
                info['propriedade_violada'] = f"Liveness (PDR colapsou para {pdr_atual:.2f}%. Original: {self.pdr_original:.2f}%)"
            
            # 2. Checagem Crítica: Safety (Atraso médio aumenta 50%+)
            elif delay_atual >= (self.delay_original * 1.5):
                recompensa = +100.0
                terminou = True
                info['propriedade_violada'] = f"Safety (Atraso médio aumentou 50%+. Original: {self.delay_original*1000:.2f}ms | Atual: {delay_atual*1000:.2f}ms)"
            
            # 3. Recompensa Parcial (Baseada em atraso de fila e perda contínua de pacotes no NS-3)
            else:
                aumento_delay = delay_atual - self.delay_original
                aumento_delay_ms = aumento_delay * 1000.0
                perda_pdr = self.pdr_original - pdr_atual

                recompensa = (aumento_delay_ms * 5.0) + ((perda_pdr / 100.0) * 10.0) - 1.0
                info['propriedade_violada'] = "Nenhuma (Ataque em andamento)"
        else:
            # Modo NetworkX
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