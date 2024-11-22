import json
import os
import pickle
import random
import torch as T

import numpy as np
import pandas as pd
import seaborn as sns
from agents.gnn_a2c import GNNA2C
from utils.env import Env
from matplotlib import pyplot as plt
from tqdm import tqdm

from agents.baselines import ConnectionWalker, DistanceWalker, GreedyWalker, RandomWalker
from utils.execution import calculate_win_rate, test
from utils.helpers import compute_ci, precompute_graph_info
import networkx as nx

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
    
    with open("results/results.json", "r") as f:
            results = json.load(f)
            
    for graph in tqdm(graphs.keys()):
        seed_dfs = []
        rw_lengths = []
        dw_lengths = []
        cw_lengths = []
        gw_lengths = []
        oracle_lengths = []
        gnn_lengths = []
        for seed in tqdm(range(5)):
            seed_df = pd.DataFrame()
            experiment_dir = os.path.join("graphs", str(graph) + f"_seed_{seed}")
            env = Env(
                max_episode_length=100,
                experiment_mode=2,
                k=1,
                experiment_dir=experiment_dir,
                pca_dim=3,
            )
            
            with open(os.path.join(experiment_dir, "data", "test_set.pkl"), "rb") as f:
                test_set = pickle.load(f)
                    
            with open(os.path.join(experiment_dir, "data", "validation_set.pkl"), "rb") as f:
                validation_set = pickle.load(f)
            
            for start, end in test_set:
                oracle_lengths.append(nx.shortest_path_length(env.g, start, end))
            
            seed_df["oracle"] = oracle_lengths
            
            # connectionwalk_scores = []
            # for temp in tqdm(softmax_temps):
            #     connection_walker = ConnectionWalker(env.g, softmax_temp=temp)
            #     connectionwalk_scores.append(
            #         np.mean(
            #             test(
            #                 connection_walker,
            #                 env,
            #                 dataset=validation_set,
            #                 seed=0,
            #             )
            #         )
            #     )
                
            # optimal_connectionwalk_temp = softmax_temps[np.argmin(connectionwalk_scores)]
            # connection_walker = ConnectionWalker(env.g, softmax_temp=optimal_connectionwalk_temp)            
            
            # distwalk_scores = []
            # for temp in tqdm(softmax_temps):
            #     dist_walker = DistanceWalker(env.g, softmax_temp=temp)
            #     distwalk_scores.append(
            #         np.mean(test(dist_walker, env, dataset=validation_set, seed=0))
            #     )

            # optimal_distwalk_temp = softmax_temps[np.argmin(distwalk_scores)]
            # dist_walker = DistanceWalker(env.g, softmax_temp=optimal_distwalk_temp)
            # random.seed(0)
            # T.manual_seed(0)
            # np.random.seed(0)
            
            # cw_lengths.extend(test(
            #         connection_walker, 
            #         env, 
            #         dataset=test_set, 
            #         seed=seed, 
            #         use_ratio=False
            #         ))
            
            # dw_lengths.extend( test(
            #         dist_walker, 
            #         env, 
            #         dataset=test_set, 
            #         seed=seed, 
            #         use_ratio=False
            #         ))
            
            # random_walker = RandomWalker(env.g)
            # rw_lengths.extend(test(
            #         random_walker, 
            #         env, 
            #         dataset=test_set, 
            #         seed=seed, 
            #         use_ratio=False
            #         ))
        
            # greedy_walker = GreedyWalker(env.g)
            # gw_lengths.extend(test(
            #         greedy_walker, 
            #         env, 
            #         dataset=test_set, 
            #         seed=seed, 
            #         use_ratio=False
            #         ))
            
            if os.path.exists(os.path.join(experiment_dir, "data", "nodes.pkl")):
                nodes = pickle.load(open(os.path.join(experiment_dir, "data", "nodes.pkl"), "rb"))

            if os.path.exists(os.path.join(experiment_dir, "data", "ego_graphs.pkl")):
                ego_graphs = pickle.load(
                    open(
                        os.path.join(experiment_dir, "data", "ego_graphs.pkl"),
                        "rb",
                    )
                )

            else:
                nodes, ego_graphs = precompute_graph_info(env.g, experiment_dir + "/data")
            
            agent = GNNA2C(
                    env=env,
                    config=json.load(open("configs/gnn.json", "r")),
                    chkpt_dir=os.path.join("graphs", f"{graph}_seed_{seed}", "models"),
                    name="gnn0",
                    device="cpu",
                    nodes=nodes,
                    ego_graphs=ego_graphs,
                    )
            agent.load_checkpoints()
            
            gnn_lengths.extend(test(agent, env, test_set, seed=seed, use_ratio=False))
        
        results[graph]["GNNA2C"] = gnn_lengths
        # results[graph] = {
        #     'oracle':oracle_lengths, 
        #     'RandomWalker':rw_lengths, 
        #     'DistanceWalker':dw_lengths, 
        #     'ConnectionWalker':cw_lengths,
        #     'GreedyWalker':gw_lengths,
        #     "GNNA2C":gnn_lengths
        #     }
        
    with open("results/results.json", "w") as f:
        json.dump(results, f)