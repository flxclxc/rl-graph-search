import argparse
import json
import math
import pickle
import random
from os import mkdir
from os.path import exists, join

import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split

try:
    from scipy.spatial import cKDTree as KDTree
except ImportError:
    _is_scipy_available = False
else:
    _is_scipy_available = True

import networkx as nx
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
        # `pair` is the pair of nodes to decide whether to join.

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
    parser = argparse.ArgumentParser(
        description="My program description"
    )

    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument(
        "--generator",
        type=str,
        choices=["kh", "ba", "ws", "fb"],
        default="kh",
    )
    parser.add_argument("--seed", type=int, default=0)

    # Generator arguments
    parser.add_argument("--n", type=int)

    # Kaiser-Hilgetag arguments
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)

    # Barabasi-Albert arguments
    parser.add_argument("--m", type=int, default=None)

    # Watts-Strogatz arguments
    parser.add_argument("--p", type=float, default=None)
    parser.add_argument("--k", type=int, default=None)

    # Facebook arguments
    parser.add_argument("--ego_graph", type=int, default=None)

    args = parser.parse_args()
    if args.experiment_name is None and args.generator != "fb":
        raise ValueError("Experiment name must be specified")

    experiment_name = args.experiment_name

    if args.generator == "kh":
        if args.beta is None:
            raise ValueError("Beta must be specified")
        if args.alpha is None:
            raise ValueError("Alpha must be specified")

        g = kaiser_hilgetag_graph(
            n=args.n, beta=args.beta, alpha=args.alpha, seed=args.seed
        )

    elif args.generator == "ba":
        if args.m is None:
            raise ValueError("m must be specified")
        if args.n is None:
            raise ValueError("n must be specified")

        g = nx.barabasi_albert_graph(
            n=args.n, m=args.m, seed=args.seed
        )

    elif args.generator == "ws":
        if args.k is None:
            raise ValueError("k must be specified")

        if args.p is None:
            raise ValueError("p must be specified")

        g = nx.watts_strogatz_graph(
            n=args.n, k=args.k, p=args.p, seed=args.seed
        )

    elif args.generator == "fb":
        data_path = "facebook_data/data/facebook/" + str(0)

        with open("facebook_data/data/facebook/0.edges") as f:
            edges = f.readlines()

        edges = [
            [int(y) for y in x.strip().split(" ")] for x in edges
        ]
        g = nx.Graph(edges)

        with open(f"{data_path}.feat") as f:
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

        experiment_name = "facebook_" + str(args.ego_graph)

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
        for _ in range(args.test_size)
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
        pos=nx.get_node_attributes(g, "pos")
        if args.generator == "kh"
        else None,
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
                "n_nodes": args.n,
                "beta": args.beta,
                "alpha": args.alpha,
            },
            file,
            indent=4,
        )
