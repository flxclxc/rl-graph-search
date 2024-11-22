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

if __name__ == '__main__':
    graphs = ["ablation_0", "ablation_1", "ablation_2", "ablation_3", "ablation_4""]
    oracle_lengths = []
    for graph in graphs:
        experiment_dir = os.path.join("graphs", graph)
        env = Env(
        max_episode_length=1000,
        experiment_mode=2,
        k=1,
        experiment_dir=experiment_dir,
        pca_dim=3,
        )
        
        with open(path.join(experiment_dir, "data", "test_set.pkl"), "rb") as f:
            test_set = pickle.load(f)

        for start, end in test_set:
            oracle_lengths.append(nx.shortest_path_length(env.g, start, end))
            
    with open("results/results.json", "r") as f:
        results = json.load(f)
    
    results['ablation']['oracle'] = oracle_lengths
    with open("results/results.json", "w") as f:
        json.dump(results, f)    