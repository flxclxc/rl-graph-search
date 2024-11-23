import argparse
import json
import os
import pickle
import random
from os import mkdir, path
from time import time

import numpy as np
import pandas as pd
import torch
import torch as T
from tqdm import tqdm

from agents.baselines import (
    ConnectionWalker,
    DistanceWalker,
    GreedyWalker,
    RandomWalker,
)
from agents.gnn_a2c import GNNA2C
from agents.mlp_a2c import MLPA2C
from utils.env import Env
from utils.execution import test
from utils.helpers import precompute_graph_info


def get_action_times(agent, env, test_set):
    action_times_cpu = []
    for start, target in tqdm(test_set):
        (
            state,
            _,
            done,
            (message, _),
        ) = env.reset(start=start, target=target)

        agent.set_target(target, message)

        while not done:
            start = time()
            action = agent.choose_action(state)
            action_times_cpu.append(1000 * (time() - start))
            state, _, done = env.step(action)

    return np.mean(action_times_cpu)


def get_runtime_results(graph):
    print(f"Running for {graph}")
    experiment_dir = path.join("graphs", graph)

    env = Env(max_episode_length=1000, k=1, experiment_dir=experiment_dir)
    nodes = pickle.load(
        open(path.join(experiment_dir, "data", "nodes.pkl"), "rb")
    )

    ego_graphs = pickle.load(
        open(
            path.join(experiment_dir, "data", "ego_graphs.pkl"),
            "rb",
        )
    )

    with open(experiment_dir + "/data/test_set.pkl", "rb") as f:
        test_set = pickle.load(f)

    with open("configs/gnn.json", "r") as f:
        config = json.load(f)

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    agent_gnn = GNNA2C(
        env=env,
        config=gnn_config,
        chkpt_dir="",
        name="GARDEN",
        device="cpu",
        nodes=nodes,
        ego_graphs=ego_graphs,
    )

    agent_mlpwd = MLPA2C(
        env=env,
        config=mlp_config,
        chkpt_dir="",
        name="MLPA2CWD",
        device="cpu",
        nodes=nodes,
        ego_graphs=ego_graphs,
        use_degree=True,
    )
    agent_mlpwd.name = "MLPA2CWD"

    agent_mlp = MLPA2C(
        env=env,
        config=mlp_config,
        chkpt_dir="",
        name="MLPA2C",
        device="cpu",
        nodes=nodes,
        ego_graphs=ego_graphs,
        use_degree=False,
    )

    greedy_walker = GreedyWalker(g=env.g)
    distance_walker = DistanceWalker(g=env.g)
    connection_walker = ConnectionWalker(g=env.g)
    random_walker = RandomWalker(g=env.g)

    result_dict = {
        "n_nodes": len(env.g.nodes()),
        "avg_num_neighbors": np.mean(
            [len(list(env.g.neighbors(node))) for node in env.g.nodes()]
        ),
    }

    start = time()
    agent_gnn.embed_ego_graphs()
    end = time()
    result_dict["Mean GARDEN Overhead"] = np.mean(
        1000 * (end - start) / len(env.g.nodes())
    )

    for agent in [
        agent_gnn,
        agent_mlpwd,
        agent_mlp,
        greedy_walker,
        distance_walker,
        connection_walker,
        random_walker,
    ]:
        result_dict[f"Mean Action Time: {agent.name}"] = get_action_times(
            agent, env, test_set
        )

    return result_dict


if __name__ == "__main__":
    T.no_grad()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gnn")
    args = parser.parse_args()

    res = []
    graphs = [
        "facebook_0",
        "facebook_348",
        "facebook_686",
        "facebook_3437",
        "facebook_414",
    ]

    gnn_config = json.load(open("configs/gnn.json", "r"))
    mlp_config = json.load(open("configs/mlp.json", "r"))

    results = pd.DataFrame(
        columns=[
            "n_nodes",
            "avg_num_neighbors",
            "Mean GARDEN Overhead",
            "Mean Action Time: GARDEN",
            "Mean Action Time: MLPA2CWD",
            "Mean Action Time: MLPA2C",
            "Mean Action Time: GreedyWalker",
            "Mean Action Time: DistanceWalker",
            "Mean Action Time: ConnectionWalker",
            "Mean Action Time: RandomWalker",
        ]
    )

    results.index.name = "Graph"

    for i, graph in enumerate(tqdm(graphs)):
        results.loc[i] = get_runtime_results(graph)

    results.sort_values(by="avg_num_neighbors").to_csv(
        "results/runtime_analysis.csv"
    )
