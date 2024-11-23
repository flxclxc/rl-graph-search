import random

import networkx as nx
import torch as T
from torch.nn import functional as F

from agents.agent import Agent


# Baseline agents
class RandomWalker(Agent):
    """
    Random walk baseline. At each step the agent chooses a random node from the
    neighbourhood graph of the current state.

    """

    def __init__(self, g: nx.graph.Graph) -> None:
        super().__init__()
        self.name = "RandomWalker"
        self.g = g

    def choose_action(self, state):
        actions = list(nx.neighbors(self.g, state))
        return random.choice(actions)


class DistanceWalker(Agent):
    """
    Hand-crafted probabilistic baseline. At each step the agent chooses a node
    from a softmax distribution over the distance between the message and the
    attributes of the nodes in the neighbourhood graph of the current state.
    """

    def __init__(
        self,
        g: nx.graph.Graph,
        softmax_temp: float = 1.0,
    ) -> None:
        super().__init__()
        self.name = "DistanceWalker"
        self.attributes = T.tensor(
            list(nx.get_node_attributes(g, "pos").values())
        )
        self.g = g
        self.softmax_temp = softmax_temp

    def choose_action(self, state: int) -> int:
        actions = list(self.g.neighbors(state))
        action_attrs = self.attributes[actions]

        dists = T.pairwise_distance(self.message, action_attrs)
        probs = F.softmax(-dists / self.softmax_temp, dim=0)

        action_idx = int(T.multinomial(probs, 1).item())
        action = actions[action_idx]

        return action


#    def set_target(self, target, message):
#       self.message = self.attributes[target]


class ConnectionWalker(Agent):
    def __init__(self, g: nx.graph.Graph, softmax_temp: float = 1.0) -> None:
        super().__init__()
        self.g = g
        self.name = "ConnectionWalker"
        self.softmax_temp = softmax_temp

    def choose_action(self, state: int) -> int:
        neighbours = list(self.g.neighbors(state))
        neighbour_degrees = [self.g.degree[int(n)] for n in neighbours]
        probs = F.softmax(
            T.tensor(neighbour_degrees) / self.softmax_temp, dim=0
        )

        action_idx = int(T.multinomial(probs, 1).item())
        return neighbours[action_idx]


class GreedyWalker(Agent):
    """
    Hand-crafted probabilistic baseline. At each step the agent chooses a node
    from a softmax distribution over the distance between the message and the
    attributes of the nodes in the neighbourhood graph of the current state.
    """

    def __init__(
        self,
        g: nx.graph.Graph,
    ) -> None:
        super().__init__()
        self.name = "GreedyWalker"
        self.g = g
        self.attributes = T.tensor(
            list(nx.get_node_attributes(g, "pos").values())
        )

    def choose_action(self, state: int) -> int:
        actions = list(self.g.neighbors(state))
        action_attrs = self.attributes[actions]

        dists = T.pairwise_distance(self.message, action_attrs)

        action_idx = dists.argmin().item()
        action = actions[action_idx]

        return action
