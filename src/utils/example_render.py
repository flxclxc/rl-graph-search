import argparse
import json
from os import path

import networkx as nx
import torch as T

from agents.gnn_a2c import GNNA2C
from agents.mlp_a2c import MLPA2C
from utils.env import Env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str)
    args = parser.parse_args()

    experiment_dir = path.join("graphs", f"{args.graph}")

    env = Env(
        max_episode_length=100,
        experiment_mode=2,
        experiment_dir=experiment_dir,
    )

    config_path = f"configs/gnn.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    agent = GNNA2C(
        env=env,
        config=config,
        chkpt_dir=path.join(experiment_dir, "models"),
        name="gnn_entropy_0",
    )

    agent.load_checkpoints()
    (
        state,
        reward,
        done,
        (message, target),
    ) = env.reset()
    env.render()

    shortest_path_length = nx.dijkstra_path_length(
        env.g, state, target
    )
    message = T.tensor(message, dtype=T.float)
    agent.set_target(target, message)
    while not done:
        state, reward, done = env.step(
            agent.choose_action(state)
        )  # message))
        env.render()

    print("Episode Length:", env.episode_length)
    print("Shortest Path Length:", shortest_path_length)
    env.save_render(path.join(experiment_dir, agent.name + ".gif"))
