import os
import pickle
from os import path

import numpy as np
from tqdm import tqdm

from agents.baselines import (ConnectionWalker, DistanceWalker)
from utils.env import Env
from utils.execution import test

if __name__ == '__main__':
    betas = [0.01, 0.05, 0.1, 0.5, 1]
    taus = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]
    graph_names = [
        "beta_0_01", 
        "beta_0p05_seed_0", 
        "beta_0_1", 
        "beta_0_5", 
        "beta_1_seed_0"
        ]
    for i, (beta, graph_name) in enumerate(zip(betas, graph_names)):
        experiment_dir = os.path.join("graphs", graph_name)
        env = Env(
            max_episode_length=100,
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

        cwalk_scores = np.zeros((len(betas), len(taus)))
        dwalk_scores = np.zeros((len(betas), len(taus)))
        
        for j, tau in enumerate(taus):
            print(tau)
            cwalker = ConnectionWalker(g=env.g, softmax_temp=tau)
            dwalker = DistanceWalker(g=env.g, softmax_temp=tau)

            cwalk_scores[i,j] = (
            np.mean(
                test(
                    cwalker,
                    env,
                    dataset=validation_set,
                    seed=0,
                )
                )
            )
            dwalk_scores[i,j] = (
            np.mean(
                test(
                    dwalker,
                    env,
                    dataset=validation_set,
                    seed=0,
                )
                )
            )
                
    import pdb;pdb.set_trace()