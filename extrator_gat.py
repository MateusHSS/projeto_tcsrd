import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch


class ExtratorFeaturesGAT(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, num_nos: int = 20):
        super().__init__(observation_space, features_dim)
        self.num_nos = num_nos
        self.num_features_no = 3

        self.gat1 = GATConv(in_channels=self.num_features_no, out_channels=16, heads=4, concat=True)
        self.gat2 = GATConv(in_channels=64, out_channels=32, heads=1, concat=False)
        self.activation = nn.ELU()

        # Projetamos de (20 nós * 32 features = 640) para features_dim.
        # Assim mantemos a identidade individual de cada nó!
        self.fc_saida = nn.Linear(self.num_nos * 32, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.size(0)
        dispositivo = observations.device
        lista_grafos = []

        for i in range(batch_size):
            x_nos = observations[i].view(self.num_nos, self.num_features_no)
            indices_vivos = (x_nos[:, 0] == 1.0).nonzero(as_tuple=True)[0]

            arestas_origem, arestas_destino = [], []
            for u in indices_vivos:
                for v in indices_vivos:
                    if u != v:
                        arestas_origem.append(u.item())
                        arestas_destino.append(v.item())

            if len(arestas_origem) == 0:
                edge_index = torch.zeros((2, 1), dtype=torch.long, device=dispositivo)
            else:
                edge_index = torch.tensor([arestas_origem, arestas_destino], dtype=torch.long, device=dispositivo)

            lista_grafos.append(Data(x=x_nos, edge_index=edge_index))

        batch_grafos = Batch.from_data_list(lista_grafos).to(dispositivo)

        h = self.gat1(batch_grafos.x, batch_grafos.edge_index)
        h = self.activation(h)
        h = self.gat2(h, batch_grafos.edge_index)
        h = self.activation(h)

        h_achatado = h.view(batch_size, self.num_nos * 32)

        saida_ppo = self.fc_saida(h_achatado)
        return saida_ppo