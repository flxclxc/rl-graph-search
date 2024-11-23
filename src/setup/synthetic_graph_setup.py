import argparse
import json
import math
import pickle
import random
from os import mkdir
from os.path import exists, join

import matplotlib.pyplot as plt
import networkx as nx
from networkx.utils import nodes_or_number, py_random_state
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split


@py_random_state(5)
@nodes_or_number(0)
def kaiser_hilgetag_graph(
    n,
    beta=0.4,
    alpha=0.1,
    domain=(0, 0, 1, 1),
    metric=None,
    seed=None,
):
    n_name, nodes = n
    n_nodes = len(nodes)
    G = nx.Graph()

    (xmin, ymin, xmax, ymax) = domain
    xcenter, ycenter = (xmin + xmax) / 2, (ymin + ymax) / 2
    G.add_node(0, pos=(xcenter, ycenter))

    # If no distance metric is provided, use Euclidean distance.
    if metric is None:
        metric = euclidean

    i = 1
    while True:
        pos_i = seed.uniform(xmin, xmax), seed.uniform(ymin, ymax)
        cands = list(range(0, i))

        pos = nx.get_node_attributes(G, "pos")

        def should_join_with(cand):
            dist = metric(pos_i, pos[cand])
            s = seed.random()
            v = beta * math.exp(-dist * alpha)
            return s < v

        nodes_to_connect = filter(should_join_with, cands)
        edges_to_add = [(i, j) for j in nodes_to_connect]

        if len(edges_to_add) > 0:
            G.add_node(i, pos=pos_i)
            G.add_edges_from(edges_to_add)
            i += 1

        if i == n_nodes:
            break

    return G


if __name__ == "__main__":
    # iterate over seeds, alpha, beta
    n = 200
    alpha = 30
    test_size = 1000

    for beta in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]:
        for seed in range(10):
            experiment_name = (
                f"kh_n_{n}_beta_{beta}_alpha_{alpha}_seed_{seed}".replace(
                    ".", "p"
                )
            )

            g = kaiser_hilgetag_graph(n=n, beta=beta, alpha=alpha, seed=seed)

            nodelist = list(g.nodes)
            train_nodes, test_nodes = train_test_split(
                nodelist, test_size=0.2
            )
            val_nodes, test_nodes = train_test_split(
                test_nodes, test_size=0.5
            )

            validation_set = [
                (random.choice(nodelist), random.choice(val_nodes))
                for _ in range(100)
            ]
            test_set = [
                (random.choice(nodelist), random.choice(test_nodes))
                for _ in range(test_size)
            ]

            if not exists("graphs"):
                mkdir("graphs")

            save_dir = join("graphs", str(experiment_name))

            if exists(save_dir):
                raise ValueError("Experiment name already exists")

            else:
                mkdir(save_dir)
                mkdir(join(save_dir, "models"))
                mkdir(join(save_dir, "episode_length_plots"))
                mkdir(join(save_dir, "data"))

            # save train and test nodes
            data_path = join(save_dir, "data")

            with open(join(data_path, "train_nodes.pkl"), "wb") as file:
                pickle.dump(train_nodes, file)
            with open(join(data_path, "validation_set.pkl"), "wb") as file:
                pickle.dump(validation_set, file)
            with open(join(data_path, "test_set.pkl"), "wb") as file:
                pickle.dump(test_set, file)

            nx.draw(
                g,
                pos=nx.get_node_attributes(g, "pos"),
                with_labels=False,
                node_size=5,
                width=0.5,
            )
            plt.savefig(join(save_dir, "graph.png"))

            with open(join(data_path, "graph.pkl"), "wb") as file:
                pickle.dump(g, file)

            with open(join(save_dir, "info.json"), "w") as file:
                json.dump(
                    {
                        "n_nodes": n,
                        "beta": beta,
                        "alpha": alpha,
                    },
                    file,
                    indent=4,
                )
