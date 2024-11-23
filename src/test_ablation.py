import json
import os
import pickle
import random

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch as T
from matplotlib import pyplot as plt
from tqdm import tqdm
import yaml

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
    # open yaml config
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    with open("results/results.json", "r") as f:
        results = json.load(f)

    with open("results/results.json", "r") as f:
        results = json.load(f)

    beta = config["experiments"]["ablation"]["beta"]
    
    print(f"Testing on beta {beta}")
    results = {}
    results["GARDEN"] = {"episode_lengths": []}
    results["MLPA2C"] = {"episode_lengths": []}
    results["MLPA2CWD"] = {"episode_lengths": []}
    results["Oracle"] = {"episode_lengths": []}

    for seed in tqdm(range(config["experiments"]["ablation"]["n_seeds"])):
        print(f"Testing on seed {seed}")
        graph = (
            f"beta_{beta}"
            f"_{seed}"
        ).replace(".", "p")
    
        random.seed(seed)
        T.manual_seed(seed)
        np.random.seed(seed)

        experiment_dir = os.path.join(
            "graphs", str(graph)
        )
        
        env = Env(max_episode_length=config['truncation_length'], k=1, experiment_dir=experiment_dir)
    
        if os.path.exists(
            os.path.join(experiment_dir, "data", "nodes.pkl")
        ):
            nodes = pickle.load(
                open(
                    os.path.join(experiment_dir, "data", "nodes.pkl"),
                    "rb",
                )
            )

        if os.path.exists(
            os.path.join(experiment_dir, "data", "ego_graphs.pkl")
        ):
            ego_graphs = pickle.load(
                open(
                    os.path.join(
                        experiment_dir, "data", "ego_graphs.pkl"
                    ),
                    "rb",
                )
            )

        else:
            nodes, ego_graphs = precompute_graph_info(
                env.g, experiment_dir + "/data"
            )

        env = Env(
            max_episode_length=config['truncation_length'],
            experiment_mode=2,
            k=1,
            experiment_dir=experiment_dir,
            pca_dim=3,
        )

        with open(
            os.path.join(experiment_dir, "data", "test_set.pkl"), "rb"
        ) as f:
            test_set = pickle.load(f)

        with open(
            os.path.join(experiment_dir, "data", "validation_set.pkl"),
            "rb",
        ) as f:
            validation_set = pickle.load(f)

        for start, end in test_set:
            results["Oracle"]["episode_lengths"].append(
                nx.shortest_path_length(env.g, start, end)
            )

        agent = GNNA2C(
            env=env,
            config=json.load(open("configs/gnn.json", "r")),
            chkpt_dir=os.path.join(experiment_dir, "models"),
            name=f"gnn0",
            device="cpu",
            nodes=nodes,
            ego_graphs=ego_graphs,
        )

        results["GARDEN"]["episode_lengths"].extend(
            test(agent, env, test_set, seed=seed, use_ratio=False)
        )

        agent = MLPA2C(
            env=env,
            config=json.load(open("configs/gnn.json", "r")),
            chkpt_dir=os.path.join(experiment_dir, "models"),
            name="mlp_entropy_" + "0",
            use_degree=True,
            device="cpu",
            nodes=nodes,
            ego_graphs=ego_graphs,
        )
        
        results["MLPA2C"]["episode_lengths"].extend(
            test(agent, env, test_set, seed=seed, use_ratio=False)
        )
        
        agent = MLPA2C(
            env=env,
            config=json.load(open("configs/gnn.json", "r")),
            chkpt_dir=os.path.join(experiment_dir, "models"),
            name="mlp_entropy_" + "0_degree",
            use_degree=True,
            device="cpu",
            nodes=nodes,
            ego_graphs=ego_graphs,
        )
        
        results["MLPA2CWD"]["episode_lengths"].extend(
            test(agent, env, test_set, seed=seed, use_ratio=False)
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
            * len([x for x in episode_lengths if x == config['truncation_length']])
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
    
    df = pd.DataFrame.from_dict(results, orient="index")
    df.index = "Agent"

    # Save the dataframe to a CSV file
    df.to_csv("results/ablation.csv")