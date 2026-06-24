import csv
import os
import argparse
import random
import networkx as nx
import numpy as np


def determine_communication_radius(num_nodes):
    """Determines the default communication radius based on the number of nodes.

    Args:
        num_nodes (int): Number of nodes in the Mesh network.

    Returns:
        float: Recommended communication radius.
    """
    if num_nodes <= 20:
        return 0.35
    elif num_nodes <= 50:
        return 0.22
    else:
        return 0.15


def generate_instances(
    num_instances, num_nodes, csv_path, communication_radius=None, seed=None
):
    """Generates a dataset of Mesh network topology instances and saves them to a CSV file.

    Args:
        num_instances (int): Number of independent networks to generate.
        num_nodes (int): Number of nodes per network.
        csv_path (str): Path of the output CSV file.
        communication_radius (float, optional): Geometric connection radius. If None, calculated automatically.
        seed (int, optional): Random seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if communication_radius is None:
        communication_radius = determine_communication_radius(num_nodes)

    dir_name = os.path.dirname(os.path.abspath(csv_path))
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["instance_id", "node_id", "x", "y", "cpu", "memory", "neighbors"]
        )

        for inst_id in range(num_instances):
            attempts = 0
            while True:
                attempts += 1
                G = nx.random_geometric_graph(num_nodes, radius=communication_radius)
                if nx.is_connected(G):
                    break
                if attempts > 1000:
                    raise RuntimeError(
                        f"Could not generate a connected network with {num_nodes} nodes and radius {communication_radius} "
                        "after 1000 attempts. Consider increasing the communication radius."
                    )

            pos = nx.get_node_attributes(G, "pos")

            for node in G.nodes():
                x, y = pos[node]
                cpu = random.uniform(0.1, 0.9)
                memory = random.uniform(0.2, 0.8)
                neighbors = ";".join(map(str, G.neighbors(node)))
                writer.writerow(
                    [
                        inst_id,
                        node,
                        f"{x:.6f}",
                        f"{y:.6f}",
                        f"{cpu:.4f}",
                        f"{memory:.4f}",
                        neighbors,
                    ]
                )

    print(f"[OK]!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generator of fixed Mesh network topology datasets for reliability tests."
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        default=100,
        help="Number of instances to generate.",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=50,
        help="Number of nodes in each network instance.",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path of the output CSV file."
    )
    parser.add_argument(
        "--radius", type=float, default=None, help="Communication radius (optional)."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )

    args = parser.parse_args()

    generate_instances(
        num_instances=args.num_instances,
        num_nodes=args.num_nodes,
        csv_path=args.output,
        communication_radius=args.radius,
        seed=args.seed,
    )
