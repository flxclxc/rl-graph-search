import json
import pickle
import random
from os import path

import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch as T
from matplotlib import pyplot as plt
from tqdm import tqdm

from agents.baselines import ConnectionWalker, DistanceWalker
from agents.gnn_a2c import GNNA2C
from utils.env import Env
from utils.execution import test


def draw_graph(
    graph,
    target,
    pos=None,
    values=None,
    ax=None,
    title=None,
    arrow_length=0.5,
):
    if ax is None:
        fig, ax = plt.subplots()

    # pos = nx.spring_layout(graph)
    if pos is None:
        pos = nx.get_node_attributes(graph, name="pos")
        if len(pos[0]) > 2:
            pos = nx.spring_layout(graph)

    nx.draw(
        graph,
        pos,
        ax=ax,
        node_size=0,
        width=0.1,
        node_color="white",
        edge_color="grey",
    )
    # make target node black
    non_target_values = [
        v for v, x in zip(values, graph.nodes()) if x != target
    ]
    if values is not None:
        jet = plt.get_cmap("jet")
        cNorm = colors.Normalize(
            vmin=min(non_target_values), vmax=max(non_target_values)
        )
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        for node in graph.nodes():
            if node == target:
                continue
            else:
                nx.draw_networkx_nodes(
                    graph,
                    pos,
                    node_size=3,
                    nodelist=[node],
                    node_color=scalarMap.to_rgba(values[node]),
                    ax=ax,
                )

        nx.draw_networkx_nodes(
            graph,
            pos,
            node_size=10,
            nodelist=[target],
            node_color="black",
            ax=ax,
        )  # scalarMap.to_rgba(max(values)), ax=ax)
        # draw edge colours as average node colour

    ax.annotate(
        "",
        xy=pos[target],
        xytext=(pos[target][0], pos[target][1] + arrow_length),
        arrowprops=dict(arrowstyle="->", color="black", lw=1),
    )
    if title is not None:
        ax.set_title(title)
    # show colour scale


if __name__ == "__main__":
    with open("config.yml", "r") as f:  
        config = yaml.safe_load(f)
          
    fb_graphs = [f"facebook_{i}" for i in config['graphs']['fb']]

    softmax_temps = config["softmax_temps"]
    
    gnn_values = []
    distwalk_values = []

    gnns = []
    optimal_dws = []

    random.seed(1)
    fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(8, 8))

    for i, graph in enumerate(fb_graphs):
        print(f"Graph: {graph}")
        experiment_dir = "graphs/" + graph
        env = Env(max_episode_length=100, k=1, experiment_dir=experiment_dir)
        pos = nx.get_node_attributes(env.g, name="pos")
        if len(pos[0]) > 2:
            pos = nx.spring_layout(env.g)

        with open("configs/gnn.json", "r") as f:
            config = json.load(f)

        agent = GNNA2C(
            env,
            config,
            chkpt_dir=experiment_dir + "/models",
            device="cpu",
            name="gnn0",
        )

        agent.load_checkpoints()

        gnns.append(agent)
        agent = gnns[i]
        with open(
            path.join(experiment_dir, "data", "test_set.pkl"), "rb"
        ) as f:
            test_set = pickle.load(f)

        with open(
            path.join(experiment_dir, "data", "validation_set.pkl"),
            "rb",
        ) as f:
            validation_set = pickle.load(f)

        print("Crossvalidating distance walker")
        # Sweep over softmax temperatures on the validation set
        distwalk_scores = []
        for temp in tqdm(softmax_temps):
            dist_walker = DistanceWalker(env.g, softmax_temp=temp)
            distwalk_scores.append(
                np.mean(
                    test(
                        dist_walker,
                        env,
                        dataset=validation_set,
                        seed=0,
                    )
                )
            )

        optimal_distwalk_temp = softmax_temps[np.argmin(distwalk_scores)]

        optimal_dws.append(optimal_distwalk_temp)
        optimal_distwalk_temp = optimal_dws[i]
        start, target = random.choice(validation_set)
        if i == 4:
            target = 50
        message = env.g.nodes[target]["pos"]
        agent.set_target(target, message)
        agent.train_mode = False

        with T.no_grad():
            gnn_val = agent.q(list(env.g.nodes()))

        attributes = T.tensor(
            list(nx.get_node_attributes(env.g, name="pos").values())
        )
        message = attributes[target]
        pairwise_dists = T.pairwise_distance(attributes, message.unsqueeze(0))
        dw_val = -pairwise_dists / optimal_distwalk_temp
        path_length_dict = nx.shortest_path_length(env.g, target)
        draw_graph(env.g, target, pos=pos, values=gnn_val, ax=ax[i, 0])
        draw_graph(env.g, target, pos=pos, values=dw_val, ax=ax[i, 1])
    
    fig.tight_layout()
    plt.savefig("results/latent_space.png", dpi=300)
