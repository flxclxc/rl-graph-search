import os
import pickle
import random
from io import BytesIO

import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch as T
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from torch.nn import functional as F

from setup.synthetic_graph_setup import kaiser_hilgetag_graph


class Env(object):
    def __init__(
        self,
        max_episode_length: int = 100,
        k: int = 1,
        seed: int = 0,
        experiment_mode: int = 2,
        generator: callable = kaiser_hilgetag_graph,
        kwargs: dict = {
            "n": 100,
            "beta": 0.4,
            "alpha": 0.1,
            "domain": (0, 0, 1, 1),
            "metric": None,
            "seed": None,
        },
        num_candidate_graphs: int = 1000,
        experiment_dir: str = None,
        curriculum_learning: bool = False,
        start_temp: float = 1e-2,
        final_temp: float = 1,
        warmup_steps: int = 20000,
        pca_dim: int = 3,
    ):
        self.max_episode_length = max_episode_length
        self.k = k
        self.n_episodes = 0
        self.seed = seed
        self.experiment_mode = experiment_mode
        self.graph_dir = os.path.join(experiment_dir, "data")
        self.curriculum_learning = curriculum_learning

        if curriculum_learning:
            self.curriculum_temp = start_temp
            self.start_temp = start_temp
            self.warmup_steps = warmup_steps
            self.final_temp = final_temp

        if experiment_mode == 1:
            self.candidate_graphs = [
                generator(**kwargs)
                for _ in range(num_candidate_graphs)
            ]
            self.g = random.choice(self.candidate_graphs)

        if experiment_mode == 2:
            with open(
                os.path.join(self.graph_dir, "graph.pkl"), "rb"
            ) as f:
                self.g = pickle.load(f)
            with open(
                os.path.join(self.graph_dir, "train_nodes.pkl"), "rb"
            ) as f:
                self.train_nodes = pickle.load(f)

        degrees = dict(self.g.degree())

        # Convert the dictionary to a list for MinMaxScaler
        degree_list = [[degrees[node]] for node in self.g.nodes()]

        # Apply Min-Max scaling to the node degrees
        scaler = MinMaxScaler()
        scaled_degrees = scaler.fit_transform(degree_list).flatten()

        # Add the scaled degrees back to the graph as an attribute
        for i, node in enumerate(self.g.nodes()):
            self.g.nodes[node]["degree"] = scaled_degrees[i]

        if len(self.g.nodes()[0]["pos"]) > 4:
            # Extract the node attributes into a matrix
            node_attributes = np.array(
                [
                    node_data["pos"]
                    for node, node_data in self.g.nodes(data=True)
                ]
            )
            node_attributes = (
                node_attributes - np.mean(node_attributes, axis=0)
            ) / (node_attributes.std() + 1e-10)
            pca = PCA(n_components=pca_dim)
            transformed_attributes = pca.fit_transform(
                node_attributes
            )

            # Add the transformed attributes back to the graph
            for i, node in enumerate(self.g.nodes()):
                self.g.nodes[node][
                    "pos_pca"
                ] = transformed_attributes[i].tolist()
            self.pos = nx.spring_layout(self.g)
        else:
            self.pos = nx.get_node_attributes(self.g, "pos")

    def update_temp(self):
        self.n_episodes += 1
        self.curriculum_temp = max(
            self.final_temp,
            self.curriculum_temp
            + (self.final_temp - self.start_temp) / self.warmup_steps,
        )

    def select_start_node(self, target):
        if self.curriculum_learning:
            self.update_temp()

            d = dict(nx.shortest_path_length(self.g, source=target))
            viable_nodes = [
                node for node in self.g.nodes() if node != target
            ]
            dists = [d[node] for node in viable_nodes]

            probs = F.softmax(
                -T.tensor(dists, dtype=T.float)
                / self.curriculum_temp,
                dim=-1,
            )
            return viable_nodes[
                T.multinomial(probs, num_samples=1).item()
            ]

        else:
            nodes = [
                node for node in self.train_nodes if node != target
            ]
            return random.choice(nodes)

    def reset_seed(self, seed: int = None):
        if seed is None:
            seed = self.seed
        random.seed(seed)

    def reset(self, start=None, target=None):
        """
        Initialise the MDP by randomly assigning a goal node.

        Returns:
            State: position of the message in the graph
            Observation: ego graph of the message position
            Reward: 1.0 if the message position is the goal node, 0.0 otherwise
            Done: True if the message position is the goal node or if the
            episode length exceeds the maximum episode length,
            False otherwise  False otherwise
            Info: goal node and message
        """
        # Initialise the MDP. For experiment 1, choose graph from list of candidates,
        # assign target/initial state randomly. For experiment 2, graph is fixed,
        # and target/initial state is chosen from a list of train/test nodes.
        if target is not None and start is not None:
            self.target = target
            self.state = start

        elif self.experiment_mode == 1:
            self.g = random.choice(self.candidate_graphs)
            self.target = random.choice(list(self.g.nodes))
            self.neighbourhood_graphs = [
                nx.ego_graph(
                    self.g, node, radius=self.k, center=False
                )
                for node in self.g.nodes()
            ]
            self.state = random.choice(
                range(self.g.number_of_nodes())
            )

        elif self.experiment_mode == 2:
            self.target = random.choice(self.train_nodes)
            self.state = self.select_start_node(self.target)

        self.message = self.g.nodes()[self.target]["pos"]
        self.render_frames = []

        reward = 0.0
        terminated = False
        self.episode_length = 0

        if self.state == self.target:
            reward = 1.0
            terminated = True

        truncated = self.episode_length >= self.max_episode_length

        self.terminated = terminated
        self.truncated = truncated

        self.action_space = list(self.g.neighbors(self.state))

        return (
            self.state,
            reward,
            terminated or truncated,
            (self.message, self.target),
        )

    def step(self, action: int):
        if self.truncated:
            raise Exception("Episode has been Truncated.")
        if self.terminated:
            raise Exception("Episode has terminated.")
        if not self.g.has_edge(self.state, action):
            raise Exception("Invalid action.")

        self.state = action
        self.episode_length += 1

        if self.episode_length >= self.max_episode_length:
            self.truncated = True

        if self.state == self.target:
            reward = 1.0
            self.terminated = True

        else:
            reward = 0.0

        self.action_space = list(self.g.neighbors(self.state))

        return (self.state, reward, self.terminated or self.truncated)

    def render(self):
        plt.ioff()
        node_colours = ["grey"] * self.g.number_of_nodes()
        node_colours[self.target] = "green"
        node_colours[self.state] = "lightblue"

        nx.draw(
            self.g,
            with_labels=False,
            arrows=False,
            node_color=node_colours,
            node_size=50,
            font_size=12,
            font_color="black",
            pos=self.pos,
        )

        buffer = BytesIO()
        plt.savefig(buffer, format="png")

        buffer.seek(0)
        self.render_frames.append(buffer)
        plt.clf()

    def save_render(self, filename, speed=300):
        imageio.mimwrite(
            filename,
            [
                imageio.v2.imread(buffer)
                for buffer in self.render_frames
            ],
            duration=speed,
            interval=speed,
        )


if __name__ == "__main__":
    env = Env(max_episode_length=25, k=2, experiment_mode=2)
    print(env.reset())
