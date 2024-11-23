import argparse
import json
import os
import pickle
import random
from os import path

import numpy as np
import torch as T

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
from utils.helpers import compute_ci

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    T.manual_seed(args.seed)

    if not os.path.exists("results"):
        os.mkdir("results")

    if not path.exists("results/results.json"):
        results = {}
    else:
        with open("results/results.json", "r") as f:
            results = json.load(f)

    if args.graph not in results.keys():
        results[args.graph] = {
            "DistanceWalker": {},
            "ConnectionWalker": {},
            "GreedyWalker": {},
            "RandomWalker": {},
            "gnn": {},
            "mlp": {},
        }

    experiment_dir = path.join("graphs", str(args.graph))

    model_dir = path.join(
        "graphs",
        args.graph,
    )

    env = Env(
        max_episode_length=100,
        experiment_mode=2,
        k=1,
        experiment_dir=experiment_dir,
        pca_dim=3,
    )

    # get training set
    dataset_dir = path.join(experiment_dir, "dataset.pkl")

    nodelist = list(env.g.nodes())

    with open(path.join(experiment_dir, "data", "test_set.pkl"), "rb") as f:
        test_set = pickle.load(f)

    if args.model == "mlp":
        with open("configs/mlp.json", "r") as f:
            config = json.load(f)
        agent = MLPA2C(
            env,
            config=config,
            chkpt_dir=os.path.join(model_dir, "models"),
            name="mlp" + str(args.seed),
        )
        agent.load_checkpoints()

    if args.model == "gnn":
        with open("configs/gnn.json", "r") as f:
            config = json.load(f)
        agent = GNNA2C(
            env,
            config=config,
            chkpt_dir=os.path.join(model_dir, "models"),
            name="gnn" + str(args.seed),
        )
        agent.load_checkpoints()

    episode_lengths = test(
        agent, env, dataset=test_set, seed=0, use_ratio=True
    )
    mean, ci = np.mean(episode_lengths), compute_ci(episode_lengths)

    if args.graph not in results:
        results[args.graph] = {}

    if args.model not in results[args.graph]:
        results[args.graph][args.model] = {}

    results[args.graph][args.model][int(args.seed)] = {
        "mean": mean,
        "ci": ci,
        "episode_lengths": episode_lengths,
    }
    print(f"Seed: {args.seed}, Episode length: {mean} +- {ci}")

    with open(
        path.join(experiment_dir, "data", "validation_set.pkl"), "rb"
    ) as f:
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

    if args.seed not in results[args.graph]["DistanceWalker"].keys():
        # Crossvalidating tau
        distwalk_scores = []
        for temp in softmax_temps:
            dist_walker = DistanceWalker(env.g, softmax_temp=temp)
            distwalk_scores.append(
                np.mean(test(dist_walker, env, dataset=validation_set))
            )
        optimal_distwalk_temp = softmax_temps[np.argmin(distwalk_scores)]
        dist_walker = DistanceWalker(
            env.g, softmax_temp=optimal_distwalk_temp
        )

        # Testing on the test set
        episode_lengths = test(dist_walker, env, dataset=test_set)
        performance, ci = np.mean(episode_lengths), compute_ci(
            episode_lengths
        )
        results[args.graph]["DistanceWalker"][args.seed] = {
            "mean": performance,
            "ci": ci,
            "episode_lengths": episode_lengths,
        }

    if args.seed not in results[args.graph]["ConnectionWalker"].keys():
        # Crossvalidating tau
        connectionwalk_scores = []
        for temp in softmax_temps:
            connection_walker = ConnectionWalker(env.g, softmax_temp=temp)
            connectionwalk_scores.append(
                np.mean(test(connection_walker, env, dataset=validation_set))
            )
        optimal_connectionwalk_temp = softmax_temps[
            np.argmin(connectionwalk_scores)
        ]
        connection_walker = ConnectionWalker(
            env.g, softmax_temp=optimal_connectionwalk_temp
        )

        # Testing on the test set
        episode_lengths = test(connection_walker, env, dataset=test_set)
        performance, ci = np.mean(episode_lengths), compute_ci(
            episode_lengths
        )
        results[args.graph]["ConnectionWalker"][args.seed] = {
            "mean": performance,
            "ci": ci,
            "episode_lengths": episode_lengths,
        }

    for i, agent in enumerate([greedy_walker, random_walker]):
        if args.seed not in results[args.graph][agent.name].keys():
            episode_lengths = test(agent, env, dataset=test_set, seed=0)
            performance, ci = np.mean(episode_lengths), compute_ci(
                episode_lengths
            )
            results[args.graph][agent.name][args.seed] = {
                "mean": performance,
                "ci": ci,
                "episode_lengths": episode_lengths,
            }

    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=4)
