import argparse
import json
import os
import pickle
import random
from os import path

import networkx as nx
import numpy as np
import torch as T
from tqdm import tqdm

from agents.baselines import (ConnectionWalker, DistanceWalker, GreedyWalker,
                              RandomWalker)
from utils.env import Env
from utils.execution import test
from utils.helpers import compute_ci

if __name__ == "__main__":
    if not path.exists("results/results.json"):
        results = {}
    else:
        with open("results/results.json", "r") as f:
            results = json.load(f)

    results_dict = {
        "oracle": [],
        "DistanceWalker": [],
        "ConnectionWalker": [],
        "GreedyWalker": [],
        "RandomWalker": [],
    }

    for seed in range(5):
        experiment_dir = os.path.join("graphs", f"ablation_{seed}")
        env = Env(
            max_episode_length=1000,
            experiment_mode=2,
            k=1,
            experiment_dir=experiment_dir,
            pca_dim=3,
        )

        # get training set
        dataset_dir = os.path.join(experiment_dir, "dataset.pkl")

        nodelist = list(env.g.nodes())

        with open(path.join(experiment_dir, "data", "test_set.pkl"), "rb") as f:
            test_set = pickle.load(f)

        with open(path.join(experiment_dir, "data", "validation_set.pkl"), "rb") as f:
            validation_set = pickle.load(f)

        random_walker = RandomWalker(env.g)
        greedy_walker = GreedyWalker(env.g)

        softmax_temps = [
            1e-4,
            5e-4,
            1e-3,
            5e-3,
            1e-2,
            5e-2,
            1e-1,
            5e-1,
            1e1,
            1e2,
        ]
        print("Crossvalidating distance walker")
        # Sweep over softmax temperatures on the validation set
        distwalk_scores = []
        for temp in tqdm(softmax_temps):
            dist_walker = DistanceWalker(env.g, softmax_temp=temp)
            distwalk_scores.append(
                np.mean(test(dist_walker, env, dataset=validation_set, seed=0))
            )

        optimal_distwalk_temp = softmax_temps[np.argmin(distwalk_scores)]
        connectionwalk_scores = []

        print("Crossvalidating connection walker")
        for temp in tqdm(softmax_temps):
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

        optimal_connectionwalk_temp = softmax_temps[np.argmin(connectionwalk_scores)]

        dist_walker = DistanceWalker(env.g, softmax_temp=optimal_distwalk_temp)
        connection_walker = ConnectionWalker(
            env.g, softmax_temp=optimal_connectionwalk_temp
        )
        agents = [
            dist_walker,
            connection_walker,
            greedy_walker,
            random_walker,
        ]

        for start, end in test_set:
            results_dict["oracle"].append(nx.shortest_path_length(env.g, start, end))

        for agent in tqdm(agents):
            random.seed(0)
            T.manual_seed(0)
            np.random.seed(0)
            results_dict[agent.name].extend(
                test(agent, env, dataset=test_set, seed=0, use_ratio=False)
            )

    for k, v in results_dict.items():
        results["ablation"][k] = v

    with open("results/results.json", "w") as f:
        json.dump(results, f)
