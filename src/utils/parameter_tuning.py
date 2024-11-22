import argparse
import json
import os
import pickle
import random
from os import path

import networkx as nx
import numpy as np
import optuna
import torch as T
from optuna.pruners import MedianPruner
from optuna.samplers import RandomSampler, TPESampler
from tqdm import tqdm

from agents.gnn_a2c import GNNA2C
from agents.mlp_a2c import MLPA2C
from utils.env import Env
from utils.execution import test
from utils.helpers import entropy


def objective(
    trial,
    seed=0,
    eval_interval=300,
):
    np.random.seed(seed)
    random.seed(seed)
    T.manual_seed(seed)

    config["lamda"] = trial.suggest_float("lamda", 0.0, 1.0)
    config["beta"] = trial.suggest_float("beta", 0.0, 1e-1)
    pbar = tqdm(range(args.n_episodes), postfix="Validation: None")

    env = Env(
        max_episode_length=50, k=1, experiment_dir=experiment_dir
    )

    agent = Agent(
        env=env,
        config=config,
        chkpt_dir=path.join(experiment_dir, "models"),
    )
    best_score = np.inf

    for i in pbar:
        agent.train()
        state, reward, done, (message, target) = env.reset()
        agent.set_target(target, message)

        states = []
        log_probs = []
        entropies = []
        message = T.tensor(
            message, device=agent.actor.device, dtype=T.float
        )

        while not done:
            states.append(state)
            probs = agent.pi(state) + 1e-10
            action_idx = T.multinomial(probs, 1).item()
            log_probs.append(T.log(probs[action_idx]))
            entropies.append(entropy(probs))

            next_state, reward, done = env.step(
                agent.nodes[state].action_list[action_idx]
            )
            state = next_state

        if len(states) > 1:
            agent.step(
                states, T.stack(log_probs), T.stack(entropies), reward
            )

        if i % eval_interval == 0:
            test_scores = test(agent, env, dataset=validation_set)
            test_score = np.mean(test_scores)
            if test_score < best_score:
                best_score = test_score
            trial.report(test_score, i)
            pbar.set_postfix(Validation=best_score)
            tqdm.refresh(pbar)

    return best_score


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph")
    parser.add_argument("--n_trials", type=int, default=15)
    parser.add_argument("--n_episodes", type=int, default=100000)
    parser.add_argument("--n_jobs", type=int, default=3)
    parser.add_argument("--model", type=str, default="mlp_a2c")

    args = parser.parse_args()
    experiment_dir = path.join("graphs", args.graph)

    with open(path.join(f"configs/{args.model}.json")) as f:
        config = json.load(f)

    with open(
        path.join(experiment_dir, "data", "validation_set.pkl"), "rb"
    ) as f:
        validation_set = pickle.load(f)

    study = optuna.create_study(
        storage=f"sqlite:///{experiment_dir}/parameter_tuning_{args.model}.db",
        study_name=f"parameter_tuning_{args.graph}_{args.model}",
        sampler=RandomSampler(),
        load_if_exists=True,
        direction="minimize",
    )

    if args.model == "mlp":
        Agent = MLPA2C

    elif args.model == "gnn":
        Agent = GNNA2C

    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )

    best_trial = study.best_trial

    # Convert best_trial to a JSON-serializable dictionary
    best_trial_dict = best_trial.params

    # Save the dictionary as JSON
    with open(
        path.join(
            experiment_dir,
            f"best_config_{args.model}.json",
        ),
        "w",
    ) as f:
        json.dump(best_trial_dict, f, indent=4)
