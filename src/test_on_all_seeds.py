import argparse
import json
import os
import pickle
import random
from os import mkdir, path

import numpy as np
import torch

from agents.gnn_a2c import GNNA2C
from agents.mlp_a2c import MLPA2C
from utils.env import Env
from utils.execution import test
from utils.helpers import precompute_graph_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str, default="facebook")
    parser.add_argument("--model", type=str, default="gnn")
    parser.add_argument("--seeds", type=list, default=None)
    args = parser.parse_args()

    experiment_dir = path.join("graphs", args.graph)

    env = Env(max_episode_length=1000, k=1, experiment_dir=experiment_dir)

    model_dir = path.join(experiment_dir, "models")

    if not path.exists(model_dir):
        mkdir(model_dir)

    if path.exists(path.join(experiment_dir, "data", "nodes.pkl")):
        nodes = pickle.load(open(path.join(experiment_dir, "data", "nodes.pkl"), "rb"))

    if path.exists(path.join(experiment_dir, "data", "ego_graphs.pkl")):
        ego_graphs = pickle.load(
            open(
                path.join(experiment_dir, "data", "ego_graphs.pkl"),
                "rb",
            )
        )

    else:
        nodes, ego_graphs = precompute_graph_info(env.g, experiment_dir + "/data")

    with open(
        path.join(experiment_dir, "data", "validation_set.pkl"),
        "rb",
    ) as f:
        validation_set = pickle.load(f)

    with open(experiment_dir + "/data/test_set.pkl", "rb") as f:
        test_set = pickle.load(f)

    with open("configs/gnn.json", "r") as f:
        config = json.load(f)

    episode_lengths = []
    if args.model == "mlp":
        Agent = MLPA2C
    elif args.model == "gnn":
        Agent = GNNA2C

    for seed in range(10):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        agent = Agent(
            env=env,
            config=json.load(open("configs/gnn.json", "r")),
            chkpt_dir=path.join("graphs", args.graph, "models"),
            name=f"{args.model}" + str(seed),
            device="cpu",
            nodes=nodes,
            ego_graphs=ego_graphs,
        )
        agent.load_checkpoints()
        results = test(agent, env, test_set, seed=seed, use_ratio=False)
        episode_lengths.extend(results)
        print(
            f"Seed: {seed}, Episode length: {np.mean(list(map(lambda x: min(1000,x), results)))}"
        )

    if not path.exists("results/results.json"):
        results = {}
    else:
        with open("results/results.json", "r") as f:
            results = json.load(f)

    truncation_rate = len([x for x in episode_lengths if x >= 100]) / len(
        episode_lengths
    )
    
    results[args.graph][
        "GNNA2C" if args.model == "gnn" else "MLPA2C With Degree"
    ] = episode_lengths

    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Truncation rate: ", truncation_rate)
    print("Mean: ", np.mean(episode_lengths))
