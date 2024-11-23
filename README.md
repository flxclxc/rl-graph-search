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
```

## Getting Started

### Prerequisites

- Python 3.10

### Clone the Repository

Clone this repository to your local machine:

```sh
git clone https://github.com/yourusername/project-name.git
cd project-name
```

### Environment Setup

Install the required Python packages using Poetry, and activate the environment:

```sh
poetry install
poetry shell
```

### Experiment Setup

To setup the real and synthetic graphs for experiments,  run the following command:

```sh
bash setup.sh

```
The graphs are stored in folder **graphs**.
### Train models

To train a model from {mlp,gnn} on a given graph, run the following command:

```sh
python src/train.py --model {MODEL} --graph {GRAPH} --n_episodes {N_EPISODES} --seed {SEED}
```
### Evaluate models

To test a model on one seed, run the following command:

```sh
python src/test.py --model {MODEL} --graph {GRAPH} --seed {SEED}

```
note: facebook graphs used were fb 0, 348, 686, 3437, 414

### Replicating Results

To replicate the results from the paper, you must train GARDEN on 10 random seeds for each facebook graph.

GARDEN must be trained on each of the synthetic graphs generated.
MLPA2C and MLPA2CWS must be trained on each of the synthetic graphs generated with $\beta=5$.

We recommend using a cluster to train the models in parallel. GARDEN may benefit from a GPU. You may need to set up a free weights and biases account to track the training progress.

Once this has been completed, the results can be generated using the scripts the src folder