import json

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from utils.execution import calculate_win_rate
from utils.helpers import compute_ci

if __name__ == "__main__":
    with open("results/results.json", "r") as f:
        results = json.load(f)

    stress_test = {}
    graphs = {
        "beta_0p01": 0.01,
        "beta_0p05": 0.05,
        "beta_0p1": 0.1,
        "beta_0p2": 0.2,
        "beta_0p3": 0.3,
        "beta_0p4": 0.4,
        "beta_0p5": 0.5,
        "beta_0p75": 0.75,
        "beta_1": 1.0,
    }
    stress_test = {graph : results[graph] for graph in graphs.keys()}

    df = pd.DataFrame(columns=["beta", "Agent", "mean", "ci"])
    fig,ax = plt.subplots(3,1,figsize=(10,10))
    for graph, result in stress_test.items():
        oracle_lengths = result["oracle"]
        for agent, lengths in result.items():
            if agent in ["oracle"]:
                continue
            
            if agent == 'GNNA2C':
                agent = 'GARDEN'
            
            lengths_ = list(map(lambda x: min(x, 100), lengths))

            oracle_ratios = [
                r if o != 0 else 1 for r, o in zip(lengths_, oracle_lengths)
            ]

            df = df._append(
                {
                    "beta": graphs[graph],
                    "Agent": agent,
                    "mean": np.mean(oracle_ratios),
                    "ci": compute_ci(oracle_ratios),
                    "Truncation Rate": 100
                    * np.mean([length >= 100 for length in lengths]),
                },
                ignore_index=True,
            )

    sns.lineplot(
        data=df,
        x="beta",
        y="mean",
        hue="Agent",
        # style="Agent",
        markers=True,
        dashes=False,
        ax=ax[0],
    )
    # add confidence intervals as "fill between" style
    for i, agent in enumerate(df.Agent.unique()):
        df_agent = df[df.Agent == agent]
        ax[0].fill_between(
            df_agent.beta,
            df_agent["mean"] - df_agent["ci"],
            df_agent["mean"] + df_agent["ci"],
            alpha=0.2,
            # ax=ax[0],
        )
    #        plt.errorbar(df_agent.beta, df_agent['mean'], yerr=df_agent['ci'], fmt='none', capsize=5, label=agent)

    # set x ticks to be the same as the betas
    #  plt.xscale('log')
    # ax[0].xticks(list(graphs.values()), list(graphs.values()))
    # set xticks on ax[0]
    ax[0].set_xticks(
        list(graphs.values()),
        list(graphs.values()), 
        rotation=30
        )
    ax[0].set_xlabel(r"$\beta$", fontsize=14)
    ax[0].set_ylabel(r"$\bar{R}_{Oracle}$", fontsize=14)
    ax[0].legend().set_visible(False)
    # plt.ylabel("Mean Episode Length")
    # plt.savefig("plots/stress_test_episode_length.png")
    # plt.close()
    # plot the same thing but with truncation rate
    sns.lineplot(
        data=df,
        x="beta",
        y="Truncation Rate",
        hue="Agent",
        # style="agent",
        markers=True,
        dashes=False,
        ax=ax[1],
        # title = r"$\bar{R}_{\text{trunc}}$",
    )
    ax[1].set_xlabel(r"$\beta$", fontsize=14)
    ax[1].set_xticks(
        list(graphs.values()),
        list(graphs.values()), 
        rotation=30
        )
    ax[1].set_ylabel(r"$\bar{R}_{trunc}}$", fontsize=14)
    ax[1].legend().set_visible(False)
    # plt.ylabel("Truncation Rate %")
    # plt.savefig("plots/stress_test_truncation.png")
    # plt.close()
    # import pdb;pdb.set_trace()
    win_rates = []
    for graph, result in stress_test.items():
        episode_lengths = []
        for agent, lengths in result.items():
            if agent in ["oracle"]:
                continue
            lengths_ = list(map(lambda x: min(x, 100), lengths))
            episode_lengths.append(lengths_)
        # import pdb;pdb.set_trace()
        win_rates.append(calculate_win_rate(np.array(episode_lengths)))
    ax[2].stackplot(
        list(graphs.values()),
        np.array(win_rates).T, #[[4,0,2,1,3]],
        labels=['RandomWalker', 'DistanceWalker', 'ConnectionWalker', 'GreedyWalker', 'GARDEN'], # ["GARDEN", "DistanceWalker", "GreedyWalker", "ConnectionWalker", "RandomWalker"],
        # colors=['C4', 'C0', 'C2', 'C1', 'C3'],
        alpha=0.6,
    )

    # Set labels and title
    ax[2].set_xticks(
        list(graphs.values()),
        list(graphs.values()), 
        rotation=30
        )
    ax[2].set_xlabel(r"$\beta$", fontsize=14)
    ax[2].set_ylabel(r"$R_{win}$", fontsize=14)
    # ax[2].set_xticks(list(graphs.values()))
    # ax[2].legend()
    # make legend colours match the stackplot
    fig.legend(
        labels=ax[2].get_legend_handles_labels()[1],
        handles=ax[2].get_legend_handles_labels()[0], 
        loc='upper center', 
        ncols=5,
        )
    # fig.legend(
    #     labels=ax[0].get_legend_handles_labels()[1], 
    #     # loc='upper center', 
    #     )
    # .title("Win Rate vs Beta")
    # plt.legend(loc="upper right")
    fig.tight_layout()
    # make it so the legend doesnt overlap with the top plot
    fig.subplots_adjust(top=0.95)
    # set font size for axis labels
    plt.savefig("plots/stress_test.png")
    plt.close()
