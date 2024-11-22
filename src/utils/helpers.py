import pickle

import networkx as nx
import numpy as np
import scipy as sp
import torch as T
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch_geometric import utils
from tqdm import tqdm


def random_argmin(array):
    min_value = np.min(array)
    min_indices = np.where(array == min_value)[0]
    random_index = np.random.choice(min_indices)
    return random_index


def calculate_grad_norm(network):
    """
    Calculates the gradient norm of a network for debugging
    """
    total_grad_norm = 0
    for param in network.parameters():
        if param.grad is None:
            continue
        param_norm = param.grad.data.norm(2)
        total_grad_norm += param_norm.item()
    return total_grad_norm


def entropy(probs):
    return -(probs * T.log(probs)).sum(-1)


def compute_ci(data, confidence=0.95):
    a = np.array(data)
    n = len(a)
    se = sp.stats.sem(a)
    h = se * sp.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return h


class Node:
    """
    Node class for storing local graph structure.
    """

    def __init__(self, ego_graph, node_idx):
        self.action_list = list(ego_graph.neighbors(node_idx))
        self.action_to_idx = {
            action: idx for idx, action in enumerate(self.action_list)
        }
        self.node_to_idx = {
            node: idx for idx, node in enumerate(ego_graph.nodes())
        }
        self.action_idxs = [
            self.node_to_idx[action] for action in self.action_list
        ]


class Node:
    def __init__(self):
        self.action_list = None
        self.action_to_idx = None
        self.node_to_idx = None
        self.action_idxs = None


def precompute_graph_info(graph, save_dir):
    """
    Precomputes local graph information for each node in the graph and saves it to disk.
    """
    print("Precomputing local graph information...")
    nodes = []
    ego_graphs = []
    for node_idx in tqdm(range(graph.number_of_nodes())):
        node = Node()
        ego_graph = nx.ego_graph(
            graph, node_idx, radius=1, center=True, undirected=True
        )
        node_list = list(ego_graph.nodes())

        node.action_list = list(ego_graph.neighbors(node_idx))
        node.action_idxs = [
            node_list.index(action) for action in node.action_list
        ]
        nodes.append(node)

        ego_graph = utils.from_networkx(ego_graph)

        feats = [ego_graph.pos]

        center_indicator = T.zeros(ego_graph.pos.shape[0], 1)
        center_indicator[node_list.index(node_idx)] = 1

        feats.append(center_indicator)

        ego_graph.x = T.cat(
            feats,
            dim=1,
        )  # add one-hot feature to indicate central node

        ego_graphs.append(ego_graph)

    pickle.dump(nodes, open(save_dir + "/nodes.pkl", "wb"))
    pickle.dump(ego_graphs, open(save_dir + "/ego_graphs.pkl", "wb"))

    return nodes, ego_graphs
