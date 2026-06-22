import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch

class GATFeaturesExtractor(BaseFeaturesExtractor):
    """Custom features extractor using Graph Attention Networks (GAT) for stable-baselines3.

    Processes flat observation vectors into graphs and runs GAT convolutions on them.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, num_nodes: int = 20):
        """Initializes the GAT feature extractor.

        Args:
            observation_space (gym.spaces.Box): The observation space of the environment.
            features_dim (int): Dimension of the output features.
            num_nodes (int): Number of nodes in the network topology.
        """
        super().__init__(observation_space, features_dim)
        self.num_nodes = num_nodes
        self.num_features_node = 3

        self.gat1 = GATConv(in_channels=self.num_features_node, out_channels=16, heads=4, concat=True)
        self.gat2 = GATConv(in_channels=64, out_channels=32, heads=1, concat=False)
        self.activation = nn.ELU()
        self.fc_output = nn.Linear(self.num_nodes * 32, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass to extract graph features from flat observations.

        Args:
            observations (torch.Tensor): Flattened observation tensor of shape (batch_size, num_nodes * 3).

        Returns:
            torch.Tensor: Feature representation tensor of shape (batch_size, features_dim).
        """
        batch_size = observations.size(0)
        device = observations.device
        graph_list = []

        for i in range(batch_size):
            x_nodes = observations[i].view(self.num_nodes, self.num_features_node)
            alive_indices = (x_nodes[:, 0] == 1.0).nonzero(as_tuple=True)[0]

            source_edges, target_edges = [], []
            for u in alive_indices:
                for v in alive_indices:
                    if u != v:
                        source_edges.append(u.item())
                        target_edges.append(v.item())

            if len(source_edges) == 0:
                edge_index = torch.zeros((2, 1), dtype=torch.long, device=device)
            else:
                edge_index = torch.tensor([source_edges, target_edges], dtype=torch.long, device=device)

            graph_list.append(Data(x=x_nodes, edge_index=edge_index))

        batch_graphs = Batch.from_data_list(graph_list).to(device)

        h = self.gat1(batch_graphs.x, batch_graphs.edge_index)
        h = self.activation(h)
        h = self.gat2(h, batch_graphs.edge_index)
        h = self.activation(h)

        h_flat = h.view(batch_size, self.num_nodes * 32)
        ppo_output = self.fc_output(h_flat)
        return ppo_output