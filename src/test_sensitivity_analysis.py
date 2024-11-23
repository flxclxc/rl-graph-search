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
from utils.env import Env
from utils.execution import calculate_win_rate, test
from utils.helpers import compute_ci, precompute_graph_info

if __name__ == "__main__":
    # open yaml config
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    with open("results/results.json", "r") as f:
        results = json.load(f)

    softmax_temps = config["softmax_temps"]

    with open("results/results.json", "r") as f:
        results = json.load(f)

    all_results = {}

    for beta in tqdm(config["experiments"]["sensitivity_analysis"]["beta"]):
        print(f"Testing on beta {beta}")
        results = {}
        results["GARDEN"] = {"episode_lengths": []}
        results["DistanceWalker"] = {"episode_lengths": []}
        results["ConnectionWalker"] = {"episode_lengths": []}
        results["GreedyWalker"] = {"episode_lengths": []}
        results["RandomWalker"] = {"episode_lengths": []}
        results["Oracle"] = {"episode_lengths": []}

        for seed in tqdm(range(config["experiments"]["sensitivity_analysis"]["n_seeds"])):
            print(f"Testing on seed {seed}")
            graph = (
                f"beta_{beta}"
                f"_seed_{seed}"
            ).replace(".", "p")
        
            random.seed(seed)
            T.manual_seed(seed)
            np.random.seed(seed)

            experiment_dir = os.path.join(
                "graphs", str(graph)
            )
            
            env = Env(max_episode_length=100, k=1, experiment_dir=experiment_dir)
        
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

            connectionwalk_scores = []
            
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
            agent = ConnectionWalker(
                env.g, softmax_temp=optimal_connectionwalk_temp
            )
            results["ConnectionWalker"]["episode_lengths"].extend(
                test(agent, env, test_set, seed=seed, use_ratio=False)
            )

            distwalk_scores = []
            for temp in tqdm(softmax_temps):
                dist_walker = DistanceWalker(env.g, softmax_temp=temp)
                distwalk_scores.append(
                    np.mean(
                        test(dist_walker, env, dataset=validation_set, seed=0)
                    )
                )

            optimal_distwalk_temp = softmax_temps[np.argmin(distwalk_scores)]
            agent = DistanceWalker(env.g, softmax_temp=optimal_distwalk_temp)
            results["DistanceWalker"]["episode_lengths"].extend(
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
        
        all_results[beta] = results
    
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

    df.index.names = ["Beta", "Agent"]

    # Save the dataframe to a CSV file
    df.to_csv("results/sensitivity_analysis.csv")
    df = df[df["Agent"] != "Oracle"]
    
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    
    sns.lineplot(
        data=df,
        x="Beta",
        y="Oracle Ratio",
        hue="Agent",
        markers=True,
        dashes=False,
        ax=ax[0],
    )
    for i, agent in enumerate(df.Agent.unique()):
        df_agent = df[df.Agent == agent]
        ax[0].fill_between(
            df_agent.beta,
            df_agent["mean"] - df_agent["ci"],
            df_agent["mean"] + df_agent["ci"],
            alpha=0.2,
        )
  
    ax[0].set_xticks(
        list(graphs.values()), list(graphs.values()), rotation=30
    )
    ax[0].set_xlabel(r"$\beta$", fontsize=14)
    ax[0].set_ylabel(r"$\bar{R}_{Oracle}$", fontsize=14)
    ax[0].legend().set_visible(False)
    
    sns.lineplot(
        data=df,
        x="beta",
        y="Truncation Rate",
        hue="Agent",
        markers=True,
        dashes=False,
        ax=ax[1],
    )
    ax[1].set_xlabel(r"$\beta$", fontsize=14)
    ax[1].set_xticks(
        list(graphs.values()), list(graphs.values()), rotation=30
    )
    ax[1].set_ylabel(r"$\bar{R}_{trunc}}$", fontsize=14)
    ax[1].legend().set_visible(False)
    
    win_rates = []
    for graph, result in stress_test.items():
        episode_lengths = []
        for agent, lengths in result.items():
            if agent in ["oracle"]:
                continue
            lengths_ = list(map(lambda x: min(x, 100), lengths))
            episode_lengths.append(lengths_)
        win_rates.append(calculate_win_rate(np.array(episode_lengths)))
    ax[2].stackplot(
        list(graphs.values()),
        np.array(win_rates).T, 
        labels=[
            "RandomWalker",
            "DistanceWalker",
            "ConnectionWalker",
            "GreedyWalker",
            "GARDEN",
        ],  
        alpha=0.6,
    )

    # Set labels and title
    ax[2].set_xticks(
        list(graphs.values()), list(graphs.values()), rotation=30
    )
    ax[2].set_xlabel(r"$\beta$", fontsize=14)
    ax[2].set_ylabel(r"$R_{win}$", fontsize=14)
   
    fig.legend(
        labels=ax[2].get_legend_handles_labels()[1],
        handles=ax[2].get_legend_handles_labels()[0],
        loc="upper center",
        ncols=5,
    )
   
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.savefig("results/sensitivity_analysis.png")
    plt.close()