# Learning to Search on Graphs
This is the code for the paper *Reinforcement Learning Discovers Efficient Decentralized Graph Path Search Strategies* by [Alexei Pisacane](https://www.linkedin.com/in/alexei-pisacane-9065141b2/), [Victor-Alexandru Darvariu](https://victor.darvariu.me), and [Mirco Musolesi](https://www.mircomusolesi.org/), presented at the Third Learning on Graphs Conference (LoG 2024). If you use this code, please consider citing our work:

```
@inproceedings{pisacane24reinforcement,
  title = {Reinforcement Learning Discovers Efficient Decentralized Graph Path Search Strategies},
  author = {Pisacane, Alexei and Darvariu, Victor-Alexandru and Musolesi, Mirco},
  booktitle = {Proceedings of the Third Learning on Graphs Conference (LoG 2024)},
  year = {2024},
  volume = {269},
  series = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
}

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
