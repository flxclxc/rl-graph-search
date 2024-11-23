import collections
import random
from os import mkdir, path

import networkx as nx
import numpy as np
import torch as T
from matplotlib import pyplot as plt
from tqdm import tqdm

import wandb

from .helpers import entropy, random_argmin


def stable_divide(a, b):
    if b != 0:
        return a / b
    elif a == 0:
        return 1
    else:
        return 1e-10


def test(agent, env, dataset, seed=None, use_ratio=True):
    if seed is not None:
        np.random.seed(seed)
        T.manual_seed(seed)
        T.cuda.manual_seed(seed)
        T.cuda.manual_seed_all(seed)
        random.seed(seed)
    episode_lengths = []

    with T.no_grad():
        agent.eval()
        for start, target in dataset:
            oracle_length = (
                nx.shortest_path_length(env.g, start, target) + 1e-10
            )
            (
                state,
                _,
                done,
                (message, _),
            ) = env.reset(start=start, target=target)
            agent.set_target(target, message)

            message = T.tensor(message, dtype=T.float)

            while not done:
                state, _, done = env.step(agent.choose_action(state))

            episode_lengths.append(
                stable_divide(env.episode_length, oracle_length)
                if use_ratio
                else env.episode_length
            )
    return episode_lengths


def train(
    agent,
    env,
    validation_set,
    experiment_dir: str,
    evaluate: bool = True,
    seed: int = 0,
    verbose: bool = True,
    eval_interval: int = 100,
    n_episodes: int = 10000,
    print_interval: int = 10,
    plot: bool = True,
    wandb_log: bool = False,
):
    np.random.seed(seed)
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    T.cuda.manual_seed_all(seed)
    random.seed(seed)

    # if wandb_log:
    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project="deterministic final step",
        # Track hyperparameters and run metadata
        config={
            "seed": seed,
            "dataset": experiment_dir.split("/")[-1],
            "deterministic_final_step": True,
        },
    )

    plot_path = path.join(
        experiment_dir,
        "episode_length_plots",
    )

    if not path.exists(plot_path):
        mkdir(plot_path)

    episode_lengths = []
    avg_episode_lengths = []
    eval_scores = []
    truncations = []
    best_eval_score = float("inf")
    truncation_rate = 1.0

    pbar = tqdm(
        range(n_episodes),
        postfix="Validation: None, Truncation: None",
    )
    for i in pbar:
        agent.train()
        state, reward, done, (message, target) = env.reset()
        agent.set_target(target, message)

        states = []
        log_probs = []
        message = T.tensor(message, device=agent.actor.device, dtype=T.float)
        oracle_length = nx.shortest_path_length(env.g, state, target) + 1e-10
        while not done:
            states.append(state)

            probs = agent.pi(state) + 1e-10
            action_idx = T.multinomial(probs, 1).item()
            log_probs.append(T.log(probs[action_idx]))

            next_state, reward, done = env.step(
                agent.nodes[state].action_list[action_idx]
            )
            state = next_state

        if len(states) > 1:
            agent.step(states, T.stack(log_probs), reward)

        truncations.append((1 - reward))
        episode_lengths.append(
            stable_divide(env.episode_length, oracle_length)
        )

        if i % eval_interval == 0 and i >= eval_interval and evaluate:
            test_scores = test(agent, env, dataset=validation_set)
            avg_test_score = np.mean(test_scores)
            if verbose:
                pbar.write(
                    f"Validation Score | Avg. Episode Length: {avg_test_score}"
                )
            tqdm.refresh(pbar)
            if avg_test_score < best_eval_score and i >= eval_interval:
                best_eval_score = avg_test_score
                agent.save_checkpoints()
                pbar.set_postfix(
                    Truncation=truncation_rate,
                    Validation=best_eval_score,
                )
                if verbose:
                    pbar.write(f"Saving model")
            eval_scores.append(avg_test_score)

        if i % print_interval == 0 and plot and i >= eval_interval:
            avg_episode_length = np.mean(episode_lengths[-eval_interval:])
            if verbose:
                pbar.write(
                    f"Episode {i} | Avg. Episode Length: {avg_episode_length}"
                )
            tqdm.refresh(pbar)
            avg_episode_lengths.append(avg_episode_length)
            truncation_rate = np.mean(truncations[-eval_interval:])
            pbar.set_postfix(
                Truncation=truncation_rate, Validation=best_eval_score
            )

            plt.plot(
                eval_interval
                + print_interval * T.arange(len(avg_episode_lengths)),
                avg_episode_lengths,
                label="Training",
            )
            plt.plot(
                eval_interval + eval_interval * T.arange(len(eval_scores)),
                eval_scores,
                label="Validation",
            )

            plt.legend()
            plt.xlabel("Episode")
            plt.ylabel("Avg. Episode Length / Oracle Length")
            plt.savefig(path.join(plot_path, agent.name + ".png"))
            plt.clf()


def trainwb(
    agent,
    env,
    validation_set,
    experiment_dir: str,
    seed: int = 0,
    eval_interval: int = 100,
    n_episodes: int = 10000,
    use_entropy=False,
    config=None,
    log=True,
    truncation_length=100,
):
    if log:
        wandb.login()
        wandb.init(
            # Set the project where this run will be logged
            project="fb_experiments",
            # Track hyperparameters and run metadata
            config={
                "seed": seed,
                "dataset": experiment_dir.split("/")[-1],
                "entropy": use_entropy,
                "beta": config["beta"],
                "truncation_length": truncation_length,
                "model": agent.name.split("_")[0],
            },
        )

    plot_path = path.join(
        experiment_dir,
        "episode_length_plots",
    )

    if not path.exists(plot_path):
        mkdir(plot_path)

    episode_lengths = collections.deque(maxlen=100)
    truncations = collections.deque(maxlen=100)
    best_eval_score = float("inf")
    truncation_rate = 100

    for i in range(n_episodes):
        agent.train()
        state, reward, done, (message, target) = env.reset()
        agent.set_target(target, message)

        states = []
        log_probs = []
        entropies = []

        message = T.tensor(message, device=agent.actor.device, dtype=T.float)
        oracle_length = nx.shortest_path_length(env.g, state, target) + 1e-10

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
            agent.step(states, T.stack(log_probs), T.stack(entropies), reward)

        truncations.append((1 - reward))
        episode_lengths.append(env.episode_length / oracle_length)
        if i % eval_interval == 0 and i >= eval_interval:
            test_scores = test(
                agent, env, dataset=validation_set, use_ratio=True
            )
            avg_test_score = np.mean(test_scores)
            if avg_test_score < best_eval_score and i >= eval_interval:
                best_eval_score = avg_test_score
                agent.save_checkpoints()

        if i % eval_interval == 0 and i >= eval_interval:
            truncation_rate = np.mean(truncations)
            if log:
                wandb.log(
                    {
                        "Truncation": truncation_rate,
                        "Validation": avg_test_score,
                        "Training": np.mean(episode_lengths),
                        "Best eval Score": best_eval_score,
                    }
                )
    if log:
        wandb.finish()


def calculate_win_rate(scores):
    wins = np.zeros(scores.shape[0])
    # compute win rate
    for trial in range(scores.shape[1]):
        win_index = random_argmin(scores[:, trial])
        wins[win_index] += 1

    win_rate = wins / scores.shape[1] * 100
    return win_rate
