import argparse
import json
import math
import pickle
import random
from os import mkdir, rmdir
from os.path import exists, join
import yaml

import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split

from networkx.utils import nodes_or_number, py_random_state


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
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    if not exists("graphs"):
        mkdir("graphs")

    if not exists("results"):
        mkdir("results")

    for idx in config["graphs"]["fb"]:
        path = f"facebook/{idx}"

        with open(f"{path}.edges", "r") as f:
            edges = f.readlines()

        edges = [[int(y) for y in x.strip().split(" ")] for x in edges]
        g = nx.Graph(edges)

        with open(f"{path}.feat") as f:
            features = f.readlines()
        features = [
            [float(x) for x in feature.split(" ")[1:]] for feature in features
        ]

        for i, node in enumerate(g.nodes()):
            g.nodes[node]["pos"] = features[i]

        largest_component = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_component)
        g = nx.convert_node_labels_to_integers(g)
        experiment_name = "facebook_" + str(idx)

        nodelist = list(g.nodes)

        train_nodes, test_nodes = train_test_split(nodelist, test_size=0.2)
        val_nodes, test_nodes = train_test_split(test_nodes, test_size=0.5)

        validation_set = [
            (random.choice(nodelist), random.choice(val_nodes))
            for _ in range(config["n_val"])
        ]
        test_set = [
            (random.choice(nodelist), random.choice(test_nodes))
            for _ in range(config["n_test"])
        ]

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
            with_labels=False,
            node_size=5,
            width=0.5,
        )

        plt.savefig(join(save_dir, "graph.png"))
        plt.close()
        with open(join(data_path, "graph.pkl"), "wb") as file:
            pickle.dump(g, file)

        with open(join(data_path, "info.json"), "w") as file:
            json.dump(
                {
                    "num_nodes": g.number_of_nodes(),
                    "num_edges": g.number_of_edges(),
                    "diameter": nx.diameter(g),
                },
                file,
            )

    for alpha in config["graphs"]["synthetic"]["alpha"]:
        for beta in config["graphs"]["synthetic"]["beta"]:
            for seed in range(config["graphs"]["synthetic"]["seeds"]):
                experiment_name = (
                    f"beta_{beta}"
                    f"_seed_{seed}".replace(".", "p")
                )

                g = kaiser_hilgetag_graph(
                    n=config["graphs"]["synthetic"]["n"],
                    beta=beta,
                    alpha=alpha,
                    seed=seed,
                )

                nodelist = list(g.nodes)
                train_nodes, test_nodes = train_test_split(
                    nodelist, test_size=0.2
                )
                val_nodes, test_nodes = train_test_split(
                    test_nodes, test_size=0.5
                )

                validation_set = [
                    (random.choice(nodelist), random.choice(val_nodes))
                    for _ in range(config["n_val"])
                ]
                test_set = [
                    (random.choice(nodelist), random.choice(test_nodes))
                    for _ in range(config["n_test"])
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
                with open(
                    join(data_path, "validation_set.pkl"), "wb"
                ) as file:
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
                            "n_nodes": config["graphs"]["synthetic"]["n"],
                            "beta": beta,
                            "alpha": alpha,
                        },
                        file,
                        indent=4,
                    )