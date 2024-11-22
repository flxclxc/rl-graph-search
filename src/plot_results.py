import argparse
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from utils.execution import calculate_win_rate, stable_divide
from utils.helpers import compute_ci

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str)
    parser.add_argument("--truncate", type=str, default="100")
    args = parser.parse_args()

    if not os.path.exists("plots"):
        os.mkdir("plots")

    with open("results/results.json", "r") as f:
        all_results = json.load(f)#[args.graph]

    graphs = all_results.keys()
    graphs = ['ablation']
    results_dfs = []
    
    for graph in graphs:
        graph_df = pd.DataFrame(
        columns=[
            "Agent", 
            "Oracle Ratio", 
            "Truncation Rate", 
            "Win Rate"
            ],
        )
        
        results = all_results[graph]
        graphs = results.keys()
        oracle_lengths = results["oracle"]

        if args.truncate == "n":
            with open(f"graphs/{args.graph}/data/info.json", "r") as f:
                tl = json.load(f)["num_nodes"]
        elif args.truncate == "100":
            tl = 100

        elif args.truncate == "diameter":
            with open(f"graphs/{args.graph}/data/info.json", "r") as f:
                tl = json.load(f)["diameter"]

        episode_lengths = []
        # map_ = {"gnn": "GNNA2C", "mlp": "MLPA2C + Degree", "mlp_no_degree": "MLPA2C"}
        data_list = []
        filtered_results = {
            key: value for key, value in results.items() if key not in ["oracle"]
        }
        
        oracle_ratios = []
        truncation_rates = []
        
        for agent, episode_lengths_ in filtered_results.items():
            truncated_episode_lengths = list(map(lambda x: min(x, tl), episode_lengths_))
            oracle_ratio = [
                r / o if o != 0 else 1
                for r, o in zip(truncated_episode_lengths, oracle_lengths)
            ]
            truncation_rate = (
                100
                * len([x for x in truncated_episode_lengths if x == tl])
                / len(truncated_episode_lengths)
            )
            
            oracle_ratios.append(
                f"{np.mean(oracle_ratio):.2f}" 
                + '±' 
                + f"{compute_ci(oracle_ratio):.2f}" 
                )
            truncation_rates.append(f"{truncation_rate:.2f}")
            
            # data_list.append(
            #     {
            #         "Agent": agent,
            #         "Mean": np.mean(truncated_episode_lengths),
            #         "CI": compute_ci(truncated_episode_lengths),
            #         "Truncation Rate": truncation_rate,
            #     }
            # )
            episode_lengths.append(episode_lengths_)

        episode_lengths = np.array(episode_lengths)
        win_rates = list(
            map(
                lambda x : f"{x:.2f}", 
                calculate_win_rate(episode_lengths)
                )
            )
        
        graph_df['Agent'] = list(filtered_results.keys())
        graph_df['Oracle Ratio'] = oracle_ratios
        graph_df['Truncation Rate'] = truncation_rates
        graph_df['Win Rate'] = win_rates
        df_tex = graph_df.to_latex(index=False)
        print(df_tex)
        graph_df['Graph'] = graph

        results_dfs.append(graph_df)
        
    results_df = pd.concat(results_dfs).round(2)
    results_df.to_csv('results/fb_results.csv', index=False)
    df_tex = results_df.to_latex(index=False)
    
        # Plot win rate as a pie chart
        # plt.tight_layout()
        # plt.figure(figsize=(6, 6))
        # sns.set(style="whitegrid")
        # winner = np.argmax(win_rates)
        # explode = [0.1 if i == winner else 0 for i in range(len(win_rates))]
        # # make gnna2c label bold
        # # labels = [map_[agent] for agent in filtered_results.keys()]
        # labels = list(filtered_results.keys())
        # pie = plt.pie(
        #     win_rates, labels=labels, explode=explode, startangle=90, autopct="%.0f%%"
        # )

        # for label, text in zip(labels, pie[1]):
        #     text.set_va("center")
        #     if label == "GNNA2C":
        #         text.set_fontweight("bold")  # Set font weight directly

        # plt.axis("equal")
        # plt.savefig(
        #     f"plots/{args.graph}_win_rate_truncation_{args.truncate}.png",
        #     bbox_inches="tight",
        # )
        # plt.close()

        # Plot oracle ratio bar chart
        # plt.tight_layout()
        # data_df = pd.DataFrame(data_list)
        # plt.figure(figsize=(10, 6))
        # ax = sns.barplot(x="Agent", y="Mean", data=data_df)
        # plt.errorbar(
        #     x=data_df["Agent"],
        #     y=data_df["Mean"],
        #     yerr=data_df["CI"],
        #     fmt="none",
        #     color="black",
        #     capsize=4,
        # )
        # for i, label in enumerate(ax.get_xticklabels()):
        #     if label.get_text() == "GNNA2C":
        #         label.set_fontweight("bold")

        # plt.xlabel("Agent")
        # plt.ylabel("Mean Episode Length / Shortest Path Length")
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.savefig(
        #     f"plots/{args.graph}_oracle_ratio_truncation_{args.truncate}.png",
        #     bbox_inches="tight",
        # )
        # plt.close()

        # # Plot oracle ratio bar chart
        # plt.tight_layout()
        # data_df = pd.DataFrame(data_list)
        # plt.figure(figsize=(10, 6))
        # ax = sns.barplot(x="Agent", y="Truncation Rate", data=data_df)

        # for i, label in enumerate(ax.get_xticklabels()):
        #     if label.get_text() == "GNNA2C":
        #         label.set_fontweight("bold")

        # plt.xlabel("Agent")
        # plt.ylabel("Truncation Rate")
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.savefig(
        #     f"plots/{args.graph}_truncation_{args.truncate}.png", bbox_inches="tight"
        # )
