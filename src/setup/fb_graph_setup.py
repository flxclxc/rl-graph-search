import argparse
import json
import math
import pickle
import random
from os import mkdir
from os.path import exists, join

import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    for idx in [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]:
        path = f"facebook/{idx}"
        with open(f"{path}.edges", "r") as f:
            edges = f.readlines()

        edges = [
            [int(y) for y in x.strip().split(" ")] for x in edges
        ]
        g = nx.Graph(edges)

        with open(f"{path}.feat") as f:
            features = f.readlines()
        features = [
            [float(x) for x in feature.split(" ")[1:]]
            for feature in features
        ]

        for i, node in enumerate(g.nodes()):
            g.nodes[node]["pos"] = features[i]

        largest_component = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_component)
        g = nx.convert_node_labels_to_integers(g)
        experiment_name = "facebook_" + str(idx)

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
            for _ in range(100)
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
