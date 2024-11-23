import argparse
import json
import os
import pickle
import random
from os import mkdir, path

import networkx as nx
import numpy as np
import pandas as pd
import torch
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
from utils.execution import calculate_win_rate, test
from utils.helpers import compute_ci, precompute_graph_info

if __name__ == "__main__":
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    graphs = configs["experiments"]["fb"]["graphs"]
    seeds = configs["experiments"]["fb"]["seeds"]

    if not os.path.exists("results"):
        os.mkdir("results")

    all_results = {}
    for graph in tqdm(graphs):
        print("Testing on", graph)
        if not os.path.exists(f"results/{graph}"):
            os.mkdir(f"results/{graph}")

        experiment_dir = path.join("graphs", graph)
        env = Env(
            max_episode_length=config["truncation_length"],
            k=1,
            experiment_dir=experiment_dir,
        )

        if path.exists(path.join(experiment_dir, "data", "nodes.pkl")):
            nodes = pickle.load(
                open(path.join(experiment_dir, "data", "nodes.pkl"), "rb")
            )

        if path.exists(path.join(experiment_dir, "data", "ego_graphs.pkl")):
            ego_graphs = pickle.load(
                open(
                    path.join(experiment_dir, "data", "ego_graphs.pkl"),
                    "rb",
                )
            )
        else:
            nodes, ego_graphs = precompute_graph_info(
                env.g, experiment_dir + "/data"
            )

        with open(
            path.join(experiment_dir, "data", "validation_set.pkl"),
            "rb",
        ) as f:
            validation_set = pickle.load(f)

        with open(experiment_dir + "/data/test_set.pkl", "rb") as f:
            test_set = pickle.load(f)

        ## Test GARDEN ##
        results = {}
        results["GARDEN"] = {"episode_lengths": []}
        results["DistanceWalker"] = {"episode_lengths": []}
        results["ConnectionWalker"] = {"episode_lengths": []}
        results["GreedyWalker"] = {"episode_lengths": []}
        results["RandomWalker"] = {"episode_lengths": []}
        results["Oracle"] = {"episode_lengths": []}

        # Crossvalidate distancewalker
        print("Crossvalidating distance walker")

        # Sweep over softmax temperatures on the validation set
        distwalk_scores = []
        for temp in tqdm(config["softmax_temps"]):
            dist_walker = DistanceWalker(env.g, softmax_temp=temp)
            distwalk_scores.append(
                np.mean(
                    test(dist_walker, env, dataset=validation_set, seed=0)
                )
            )

        optimal_distwalk_temp = config["softmax_temps"][
            np.argmin(distwalk_scores)
        ]

        connectionwalk_scores = []
        print("Crossvalidating connection walker")
        for temp in tqdm(config["softmax_temps"]):
            connection_walker = ConnectionWalker(env.g, softmax_temp=temp)
            connectionwalk_scores.append(
                np.mean(
                    test(
                        connection_walker,
                        env,
                        dataset=validation_set,
                        seed=0,
                    )
                )
            )

        optimal_connectionwalk_temp = config["softmax_temps"][
            np.argmin(connectionwalk_scores)
        ]

        for seed in tqdm(seeds):
            print(f"Seed: {seed}")
            model_dir = path.join(experiment_dir, "models", "gnn" + str(seed))

            if not path.exists(model_dir):
                continue

            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)

            with open("configs/gnn.json", "r") as f:
                config = json.load(f)

            agent = GNNA2C(
                env=env,
                config=json.load(open("configs/gnn.json", "r")),
                chkpt_dir=path.join(experiment_dir, "models"),
                name=f"gnn{seed}",
                device="cpu",
                nodes=nodes,
                ego_graphs=ego_graphs,
            )

            agent.load_checkpoints()
            results["GARDEN"]["episode_lengths"].extend(
                test(agent, env, test_set, seed=seed, use_ratio=False)
            )

            agent = DistanceWalker(env.g, softmax_temp=optimal_distwalk_temp)
            results["DistanceWalker"]["episode_lengths"].extend(
                test(agent, env, test_set, seed=seed, use_ratio=False)
            )

            agent = ConnectionWalker(
                env.g, softmax_temp=optimal_connectionwalk_temp
            )
            results["ConnectionWalker"]["episode_lengths"].extend(
                test(agent, env, test_set, seed=seed, use_ratio=False)
            )

            agent = RandomWalker(env.g)
            results["RandomWalker"]["episode_lengths"].extend(
                test(agent, env, test_set, seed=seed, use_ratio=False)
            )

            agent = GreedyWalker(env.g)
            results["GreedyWalker"]["episode_lengths"].extend(
                test(agent, env, test_set, seed=seed, use_ratio=False)
            )

            for start, end in test_set:
                results["Oracle"]["episode_lengths"].append(
                    nx.shortest_path_length(env.g, start, end)
                )

        for agent, res in results.items():

            episode_lengths = res["episode_lengths"]
            oracle_ratios = [
                x / y if y != 0 else 1
                for x, y in zip(
                    episode_lengths, results["Oracle"]["episode_lengths"]
                )
            ]

            results[agent]["Oracle Ratio"] = (
                f"{np.mean(oracle_ratios):.2f}"
                + "±"
                + f"{compute_ci(oracle_ratios):.2f}"
            )

            results[agent]["Truncation Rate"] = (
                100
                * len([x for x in episode_lengths if x == 100])
                / len(episode_lengths)
            )

        # compute win rate
        episode_lengths = np.array(
            [
                res["episode_lengths"]
                for agent, res in results.items()
                if agent != "Oracle"
            ]
        )
        win_rates = list(
            map(lambda x: f"{x:.2f}", calculate_win_rate(episode_lengths))
        )

        for agent, win_rate in zip(results.keys(), win_rates):
            results[agent]["Win Rate"] = win_rate

        all_results[graph] = results

    # Convert results to a double indexed pandas dataframe
    df = pd.DataFrame.from_dict(
        {
            (i, j): all_results[i][j]
            for i in all_results.keys()
            for j in all_results[i].keys()
        },
        orient="index",
    )

    df = df[
        ["Oracle Ratio", "Truncation Rate", "Win Rate", "episode_lengths"]
    ]

    df.index.names = ["Graph", "Agent"]

    # Save the dataframe to a CSV file
    df.to_csv("results/fb.csv")
