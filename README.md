# Learning to Search on Graphs

Graph path search is a classic computer science problem that has been recently approached with Reinforcement Learning (RL) due to its potential to outperform prior methods. Existing RL techniques typically assume a global view of the network, which is not suitable for large-scale, dynamic, and privacy-sensitive settings. An area of particular interest is search in social networks due to its numerous applications. 
Inspired by seminal work in experimental sociology, which showed that decentralized yet efficient search is possible in social networks, we frame the problem as a collaborative task between multiple agents equipped with a limited local view of the network. We propose a multi-agent approach for graph path search that successfully leverages both homophily and structural heterogeneity. Our experiments, carried out over synthetic and real-world social networks, demonstrate that our model significantly outperforms learned and heuristic baselines. Furthermore, our results show that meaningful embeddings for graph navigation can be constructed using reward-driven learning.

## Getting Started

### Prerequisites

- Python 3.10

### Clone the Repository

Clone this repository to your local machine:

```sh
git clone https://github.com/yourusername/project-name.git
cd project-name
```

### Install the Requirements

Install the required Python packages using Poetry:

```sh
poetry install
```

### Download the Data

Download the Facebook data to your local machine, and run the experimental setup script to preprocess the graphs:

```sh
bash download_fb_data.sh
python src/setup/fb_graph_setup.py
```
### Create synthetic graphs

To generate a synthetic graph with parameters, run the following command:
```sh
python src/setup/synthetic_graph_setup --n {N} --alpha {ALPHA} --beta {BETA} --experiment_name {EXPERIMENT_NAME} --seed {SEED}
```
### Train models

To train a model from {mlp,gnn} on a given graph, run the following command:

```sh
python src/train.py --model {MODEL} --graph {GRAPH} --n_episodes {N_EPISODES} --seed {SEED}
```
### Evaluate models

To test a model on one seed, run the following command:

```sh
python src/test.py --model {MODEL} --graph {GRAPH} --seed {SEED}


note: facebook graphs used were fb 0, 348, 686, 3437, 414
