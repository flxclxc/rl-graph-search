import os

import networkx as nx
import torch as T
from torch_geometric import utils
from tqdm import tqdm

from agents.agent import Agent
from utils.helpers import Node


class A2C(Agent):
    def __init__(
        self,
        env,
        config,
        use_ego_graph,
        use_pca=False,
        ego_graphs=None,
        nodes=None,
        use_entropy=False,
    ):
        # Collect local graph information
        self.train_mode = True
        self.greedy_mode = False

        if nodes is None and ego_graphs is None:
            nodes = []
            ego_graphs = []
            for node_idx in tqdm(range(env.g.number_of_nodes())):
                node = Node()
                ego_graph = nx.ego_graph(
                    env.g,
                    node_idx,
                    radius=1,
                    center=True,
                    undirected=True,
                )
                node_list = list(ego_graph.nodes())

                node.action_list = list(ego_graph.neighbors(node_idx))
                node.action_idxs = [
                    node_list.index(action)
                    for action in node.action_list
                ]
                nodes.append(node)

                if use_ego_graph:
                    ego_graph = utils.from_networkx(ego_graph)
                    feats = [
                        ego_graph.pos_pca
                        if use_pca
                        else ego_graph.pos
                    ]

                    center_indicator = T.zeros(
                        ego_graph.pos.shape[0], 1
                    )
                    center_indicator[node_list.index(node_idx)] = 1

                    feats.append(center_indicator)

                    ego_graph.x = T.cat(
                        feats,
                        dim=1,
                    )  # add one-hot feature to indicate central node

                    ego_graphs.append(ego_graph)

        self.nodes = nodes
        self.ego_graphs = ego_graphs
        self.use_entropy = use_entropy

        # Hyperparameters
        self.gamma = config["gamma"]
        self.beta = config["beta"]

    def pi(self, state):
        attributes = self.attributes[
            self.nodes[state].action_list
        ].to(self.actor.device)  # (n_actions, attribute_dim)

        message = self.message.unsqueeze(0).expand(
            attributes.shape[0], -1
        )  # (n_actions, message_dim)
        
        return self.actor.forward(
            T.cat((attributes, message), dim=-1)
        )

    def choose_action(self, state):
        with T.no_grad():
            probs = self.pi(state)

        if self.greedy_mode:
            action_idx = T.argmax(probs).item()
        else:
            action_idx = T.multinomial(probs, 1).item()

        return self.nodes[state].action_list[action_idx]

    def q(self, states):
        """
        Args:
            states (list): (n_states,)
            message (T.tensor): (message_dim,)

        Returns:
            values (T.tensor): (n_states,)
        """

        attributes = self.attributes[states]
        message = self.message.unsqueeze(0).expand(
            attributes.shape[0], -1
        )

        return self.critic(
            T.cat((message, attributes), dim=-1)
        ).squeeze(-1)

    def step(self, states, log_probs, entropies, final_reward):
        values = self.q(states)
        returns = T.cat(
            (
                self.gamma * values[1:],
                T.tensor([final_reward], device=self.actor.device),
            )
        )

        advantage = returns - values
        critic_loss = advantage.pow(2).mean()

        actor_loss = -(log_probs * advantage.detach()).mean()
        if self.use_entropy:
            actor_loss -= self.beta * entropies.mean()

        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False
