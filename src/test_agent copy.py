import argparse
import json
import os
import pickle
import random
from os import path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch as T

from agents.baselines import (ConnectionWalker, DistanceWalker, GreedyWalker,
                              RandomWalker)
from agents.gnn_a2c import GNNA2C
from agents.mlp_a2c import MLPA2C
from utils.env import Env
from utils.execution import calculate_win_rate, test
from utils.helpers import compute_ci

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    
    random.seed(args.seed)
    np.random.seed(args.seed)
    T.manual_seed(args.seed)
    
    if not os.path.exists("results"):
        os.mkdir("results")
        
    if not path.exists("results/resultsdir.json"):
        results = {}
    else:
        with open("results/resultsdir.json", "r") as f:
            results = json.load(f)
            
    experiment_dir = path.join("graphs", str(args.graph))
    
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

    

