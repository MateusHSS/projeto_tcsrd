import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import os
import subprocess
import json
import shutil

def load_env():
    """Lê variáveis de configuração do arquivo .env.

    Returns:
        dict: Dicionário contendo as configurações carregadas.
    """
    config = {"USE_NS3": "False", "NS3_PATH": "/home/user/ns-3.48"}
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

class FaultInjectionEnvironment(gym.Env):
    """Ambiente de simulação de injeção de falhas em redes Mesh para aprendizado por reforço.
    
    Suporta simulações via NetworkX ou físicas usando o simulador NS-3.
    """
    def __init__(self, num_nodes=20, use_ns3=None, ns3_path=None, instances_path=None):
        """Inicializa o ambiente de injeção de falhas.

        Args:
            num_nodes (int): Número de nós da rede Mesh.
            use_ns3 (bool, optional): Se True, utiliza o NS-3 para simulação de pacotes físicos.
            ns3_path (str, optional): Caminho do diretório de instalação do NS-3.
            instances_path (str, optional): Caminho do CSV de instâncias de rede pré-geradas.
        """
        super(FaultInjectionEnvironment, self).__init__()
        self.num_nodes = num_nodes

        env_config = load_env()
        
        if use_ns3 is None:
            self.use_ns3 = env_config.get("USE_NS3", "False").lower() in ("true", "1", "yes")
        else:
            self.use_ns3 = use_ns3

        if self.use_ns3:
            if ns3_path is None:
                self.ns3_path = env_config.get("NS3_PATH")
            else:
                self.ns3_path = ns3_path

            if not self.ns3_path or self.ns3_path == "/home/user/ns-3.48":
                raise ValueError("NS3_PATH must be explicitly configured in the environment or .env file when use_ns3=True.")
            
            if not os.path.exists(self.ns3_path):
                raise FileNotFoundError(f"The specified NS3_PATH does not exist: {self.ns3_path}")
        else:
            self.ns3_path = ns3_path or env_config.get("NS3_PATH", "/home/user/ns-3.48")

        if self.num_nodes <= 20:
            self.communication_radius = 0.35
            self.step_limit = 15
        elif self.num_nodes <= 50:
            self.communication_radius = 0.22
            self.step_limit = 35
        else:
            self.communication_radius = 0.15
            self.step_limit = int(self.num_nodes * 0.7)

        if self.use_ns3:
            self._copy_simulation_script()

        self.action_space = gym.spaces.Discrete(self.num_nodes)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_nodes * 3,),
            dtype=np.float32
        )

        self.graph_ptr = None
        self.steps_taken = 0
        self.failed_nodes = set()

        if instances_path is None:
            self.instances_path = env_config.get("INSTANCES_PATH", None)
        else:
            self.instances_path = instances_path

        self.instances = None
        if self.instances_path:
            self.instances = self._load_instances_csv(self.instances_path)
            if self.instances:
                first_inst_id = list(self.instances.keys())[0]
                num_nodes_in_file = len(self.instances[first_inst_id])
                if num_nodes_in_file != self.num_nodes:
                    raise ValueError(
                        f"Incompatibilidade de tamanho de rede: o ambiente foi configurado para {self.num_nodes} nós, "
                        f"mas o arquivo '{self.instances_path}' contém instâncias com {num_nodes_in_file} nós."
                    )

    def _load_instances_csv(self, csv_path):
        """Carrega as instâncias de topologia Mesh a partir de um arquivo CSV.

        Args:
            csv_path (str): Caminho para o arquivo CSV de instâncias.

        Returns:
            dict: Dicionário mapeando IDs de instâncias para seus respectivos nós e conexões.
        """
        import csv
        instances = {}
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Arquivo de instâncias não encontrado em: {csv_path}")
            
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                inst_id = int(row['instance_id'])
                node_id = int(row['node_id'])
                x = float(row['x'])
                y = float(row['y'])
                cpu = float(row['cpu'])
                memory = float(row['memory'])
                
                neighbors_str = row['neighbors'].strip()
                neighbors = [int(n) for n in neighbors_str.split(';') if n.strip()] if neighbors_str else []
                
                if inst_id not in instances:
                    instances[inst_id] = []
                instances[inst_id].append({
                    'node_id': node_id,
                    'x': x,
                    'y': y,
                    'cpu': cpu,
                    'memory': memory,
                    'neighbors': neighbors
                })
        return instances

    def _copy_simulation_script(self):
        """Garante a presença do script de simulação C++ no diretório scratch do NS-3."""
        diretorio_atual = os.path.dirname(os.path.abspath(__file__))
        src_cc = os.path.join(diretorio_atual, "mesh_simulation.cc")
        dest_cc = os.path.join(self.ns3_path, "scratch", "mesh_simulation.cc")

        if os.path.exists(src_cc):
            if not os.path.exists(dest_cc) or os.path.getmtime(src_cc) > os.path.getmtime(dest_cc):
                print(f"[NS3-Bridge] Copiando {src_cc} para {dest_cc}...")
                os.makedirs(os.path.dirname(dest_cc), exist_ok=True)
                shutil.copy(src_cc, dest_cc)
        else:
            print(f"[NS3-Bridge] Aviso: {src_cc} não encontrado localmente no projeto.")

    def _run_ns3_simulation(self):
        """Executa o simulador NS-3 em subprocesso e retorna as métricas coletadas.

        Returns:
            dict: Métricas de PDR, atraso médio e contagem de pacotes.
        """
        failed_nodes_str = ",".join(map(str, self.failed_nodes)) if self.failed_nodes else "none"
        
        cpus = [self.G.nodes[i]['features'][0] for i in range(self.num_nodes)]
        mems = [self.G.nodes[i]['features'][1] for i in range(self.num_nodes)]
        
        cpu_str = ",".join(f"{c:.4f}" for c in cpus)
        mem_str = ",".join(f"{m:.4f}" for m in mems)

        pos = nx.get_node_attributes(self.G, 'pos')
        posicoes_lista = []
        for i in range(self.num_nodes):
            x, y = pos[i]
            posicoes_lista.append(f"{x * 500.0:.2f},{y * 500.0:.2f}")
        pos_str = ",".join(posicoes_lista)

        range_fisico = self.communication_radius * 500.0

        comando = [
            "./ns3", "run", "scratch/mesh_simulation",
            "--",
            f"--numNodes={self.num_nodes}",
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
                check=True,
                timeout=60.0
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

        except subprocess.TimeoutExpired as e:
            print(f"[NS3-Bridge] Tempo limite atingido (timeout) ao executar o simulador NS-3: {e}")
            return {"pdr": 0.0, "avg_delay": 9.99, "tx_packets": 0, "rx_packets": 0}
        except (subprocess.CalledProcessError, ValueError) as e:
            print(f"[NS3-Bridge] Falha na execução do simulador: {e}")
            return {"pdr": 0.0, "avg_delay": 9.99, "tx_packets": 0, "rx_packets": 0}

    def reset(self, seed=None, options=None):
        """Reinicia o ambiente para um novo episódio de simulação.

        Args:
            seed (int, optional): Semente aleatória para o gerador Gymnasium.
            options (dict, optional): Parâmetros adicionais, como 'instancia_id'.

        Returns:
            tuple: Observação inicial e dicionário de informações adicionais.
        """
        super().reset(seed=seed)
        self.failed_nodes.clear()
        self.steps_taken = 0

        if self.instances:
            inst_id = None
            if options and 'instancia_id' in options:
                inst_id = options['instancia_id']
            else:
                inst_id = int(self.np_random.choice(list(self.instances.keys())))
                
            nos_dados = self.instances[inst_id]
            self.G = nx.Graph()
            
            for nd in nos_dados:
                node_id = nd['node_id']
                self.G.add_node(
                    node_id, 
                    status=1.0, 
                    features=[nd['cpu'], nd['memory']], 
                    pos=(nd['x'], nd['y'])
                )
                
            for nd in nos_dados:
                node_id = nd['node_id']
                for vizinho in nd['neighbors']:
                    self.G.add_edge(node_id, vizinho)
        else:
            while True:
                self.G = nx.random_geometric_graph(self.num_nodes, radius=self.communication_radius)
                if nx.is_connected(self.G):
                    break

            for node in self.G.nodes():
                self.G.nodes[node]['status'] = 1.0
                self.G.nodes[node]['features'] = [
                    self.np_random.uniform(0.1, 0.9),
                    self.np_random.uniform(0.2, 0.8)
                ]

        if self.use_ns3:
            dados_saudaveis = self._run_ns3_simulation()
            self.original_pdr = dados_saudaveis.get("pdr", 100.0)
            self.original_delay = dados_saudaveis.get("avg_delay", 0.001)
            if self.original_delay == 0.0:
                self.original_delay = 0.001
        else:
            self.original_latency = nx.average_shortest_path_length(self.G)

            if self.num_nodes <= 20:
                self.original_redundancy = nx.average_node_connectivity(self.G)
            else:
                self.original_redundancy = nx.node_connectivity(self.G)

        return self._get_observation(), {}

    def _get_observation(self):
        """Converte a topologia atual do grafo NetworkX em um vetor de observação.

        Returns:
            np.ndarray: Array contendo o status e métricas de todos os nós.
        """
        obs = []
        for i in range(self.num_nodes):
            status = self.G.nodes[i]['status']
            cpu, mem = self.G.nodes[i]['features']
            obs.extend([status, cpu, mem])
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """Executa uma ação de ataque (desativação de nó) no ambiente.

        Args:
            action (int): Índice do nó a ser atacado.

        Returns:
            tuple: Nova observação, recompensa, flag de término, flag de truncamento e metadados.
        """
        self.steps_taken += 1
        recompensa = 0.0
        terminou = False
        truncou = False

        if self.steps_taken >= self.step_limit:
            truncou = True

        if self.G.nodes[action]['status'] == 0.0:
            recompensa = -5.0
            return self._get_observation(), recompensa, terminou, truncou, {}

        self.G.nodes[action]['status'] = 0.0

        if self.use_ns3:
            self.failed_nodes.add(action)
            dados = self._run_ns3_simulation()
            pdr_atual = dados["pdr"]
            delay_atual = dados["avg_delay"]
            rx_packets = dados.get("rx_packets", 0)

            if rx_packets == 0 or pdr_atual == 0.0:
                delay_atual = 9.99

            info = {}

            if pdr_atual <= (self.original_pdr * 0.1):
                recompensa = +100.0
                terminou = True
                info['propriedade_violada'] = f"Liveness (PDR colapsou para {pdr_atual:.2f}%. Original: {self.original_pdr:.2f}%)"
            
            elif delay_atual >= (self.original_delay * 1.5):
                recompensa = +100.0
                terminou = True
                info['propriedade_violada'] = f"Safety (Atraso médio aumentou 50%+. Original: {self.original_delay*1000:.2f}ms | Atual: {delay_atual*1000:.2f}ms)"
            
            else:
                aumento_delay = delay_atual - self.original_delay
                aumento_delay_ms = aumento_delay * 1000.0
                perda_pdr = self.original_pdr - pdr_atual

                recompensa = (aumento_delay_ms * 5.0) + ((perda_pdr / 100.0) * 10.0) - 1.0
                info['propriedade_violada'] = "Nenhuma (Ataque em andamento)"
        else:
            arestas_para_remover = list(self.G.edges(action))
            self.G.remove_edges_from(arestas_para_remover)

            nos_vivos = [n for n in self.G.nodes() if self.G.nodes[n]['status'] == 1.0]
            subgrafo = self.G.subgraph(nos_vivos)

            info = {}

            if not nx.is_connected(subgrafo):
                recompensa = +100.0
                terminou = True
                info['propriedade_violada'] = "Liveness (Conectividade Rompida - Rede Particionada)"
            else:
                latencia_atual = nx.average_shortest_path_length(subgrafo)

                if self.num_nodes <= 20:
                    redundancia_atual = nx.average_node_connectivity(subgrafo)
                else:
                    redundancia_atual = nx.node_connectivity(subgrafo)

                if latencia_atual >= (self.original_latency * 1.5):
                    recompensa = +100.0
                    terminou = True
                    info['propriedade_violada'] = f"Safety (Latência aumentou 50%+. Original: {self.original_latency:.2f} | Atual: {latencia_atual:.2f})"
                else:
                    aumento_latencia = latencia_atual - self.original_latency
                    perda_redundancia = self.original_redundancy - redundancia_atual

                    recompensa = (aumento_latencia * 5.0) + (perda_redundancia * 2.0) - 1.0
                    info['propriedade_violada'] = "Nenhuma (Ataque em andamento)"

        return self._get_observation(), recompensa, terminou, truncou, info
