import argparse
import json
import pickle
import random
from os import mkdir, path

import numpy as np
import torch as T
import yaml

from agents.gnn_a2c import GNNA2C
from agents.mlp_a2c import MLPA2C
from utils.env import Env
from utils.execution import trainwb
from utils.helpers import precompute_graph_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str)
    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument("--model", type=str, default="gnn")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_log", action="store_false")
    parser.add_argument("--n_episodes", type=int, default=100000)

    args = parser.parse_args()
    experiment_dir = path.join("graphs", str(args.graph))
    seed = args.seed

    # open yaml config
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    np.random.seed(seed)
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    T.cuda.manual_seed_all(seed)
    random.seed(seed)

    env = Env(
        max_episode_length=config["truncation_length"],
        k=1,
        experiment_dir=experiment_dir,
    )

    print("env loaded")
    model_dir = path.join(experiment_dir, "models")
    if not path.exists(model_dir):
        mkdir(model_dir)

    if args.model == "gnn":
        Agent = GNNA2C

    if args.model == "mlp":
        Agent = MLPA2C

    print(f"Loading config from {config_path}")

    if path.exists(path.join(experiment_dir, "data", "nodes.pkl")):
        nodes = pickle.load(
            open(path.join(experiment_dir, "data", "nodes.pkl"), "rb")
        )

    if path.exists(path.join(experiment_dir, "data", "ego_graphs.pkl")):
        ego_graphs = pickle.load(
            open(
                path.join(experiment_dir, "data", "ego_graphs.pkl"),
                "rb",
            )
        )

    else:
        nodes, ego_graphs = precompute_graph_info(
            env.g, experiment_dir + "/data"
        )

    agent = Agent(
        env=env,
        config=config[args.model],
        chkpt_dir=path.join(experiment_dir, "models"),
        device=("cuda:0" if T.cuda.is_available() else "cpu")
        if args.cuda
        else "cpu",
        nodes=nodes,
        use_degree=True,
        ego_graphs=ego_graphs,
        name=args.model + str(args.seed),
    )

    print(f"Using device {agent.actor.device}")
    print(f"Model: {agent.name}")
    if args.load_checkpoint:
        agent.load_checkpoints()

    with open(
        path.join(experiment_dir, "data", "validation_set.pkl"),
        "rb",
    ) as f:
        validation_set = pickle.load(f)

    trainwb(
        agent,
        env,
        validation_set,
        experiment_dir,
        n_episodes=args.n_episodes,
        seed=args.seed,
        use_entropy=True,
        config=config,
        log=args.no_log,
        truncation_length=args.truncation_length,
    )
