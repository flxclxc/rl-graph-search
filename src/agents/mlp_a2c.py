import os

import networkx as nx
import torch as T

from agents.a2c import A2C
from agents.networks import ActorNetwork, CriticNetwork


class MLPA2C(A2C):
    def __init__(
        self,
        env,
        config,
        chkpt_dir,
        name="a2c",
        device="cpu",
        use_degree=True,
        nodes=None,
        ego_graphs=None,
    ):
        super().__init__(
            env,
            config,
            use_ego_graph=False,
            use_pca=False,
            nodes=nodes,
            ego_graphs=ego_graphs,
        )
        self.use_degree = use_degree
        feat = (
            "pos_pca"
            if "pos_pca" in env.g.nodes()[0].keys()
            else "pos"
        )
        self.attributes = T.tensor(
            list(nx.get_node_attributes(env.g, feat).values()),
            dtype=T.float32,
        )
        self.name = name
        self.message_dim = self.attributes.shape[1]
        if use_degree:
            degrees = T.tensor(
                list(
                    nx.get_node_attributes(env.g, "degree").values()
                ),
                dtype=T.float32,
            )
            self.attributes = T.cat(
                (self.attributes, degrees.unsqueeze(-1)), dim=-1
            )
            self.name += "WD"

        self.save_dir = os.path.join(chkpt_dir, self.name)

        self.attribute_dim = self.attributes.shape[1]
        self.gamma = config["gamma"]

        self.actor = ActorNetwork(
            in_dim=self.attribute_dim + self.message_dim,
            h_dim=config["h_actor"],
            chkpt_dir=self.save_dir,
            name="actor",
            device=device,
        )

        self.critic = CriticNetwork(
            in_dim=self.attribute_dim + self.message_dim,
            h_dim=config["h_critic"],
            chkpt_dir=self.save_dir,
            name="critic",
            device=device,
        )

        self.attributes = self.attributes.to(self.actor.device)

        parameters = list(self.actor.parameters()) + list(
            self.critic.parameters()
        )
        self.optimizer = T.optim.Adam(parameters, lr=config["lr"])

    def set_target(self, target, message):
        if self.use_degree:
            self.message = self.attributes[target][:-1]
        else:
            self.message = self.attributes[target]

    def choose_action(self, state):
        with T.no_grad():
            probs = self.pi(state)

        action_idx = T.multinomial(probs, 1).item()

        return self.nodes[state].action_list[action_idx]

    def q(self, states):
        """
        Args:
            states (list): (n_states,)
            message (T.tensor): (message_dim,)

        Returns:
            values (T.tensor): (n_states,)
        """

        attributes = self.attributes[states]
        message = self.message.unsqueeze(0).expand(
            attributes.shape[0], -1
        )

        return self.critic(
            T.cat((message, attributes), dim=-1)
        ).squeeze(-1)

    def save_checkpoints(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_checkpoints(self, load_dir=None):
        if load_dir is not None:
            self.actor.load_checkpoint(
                os.path.join(load_dir, "models", self.name, "actor")
            )
            self.critic.load_checkpoint(
                os.path.join(load_dir, "models", self.name, "critic")
            )
        else:
            self.actor.load_checkpoint()
            self.critic.load_checkpoint()
        self.actor.train()

    def save_checkpoints(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_checkpoints(self, load_dir=None):
        if load_dir is not None:
            self.actor.load_checkpoint(
                os.path.join(load_dir, "models", self.name, "actor")
            )
            self.critic.load_checkpoint(
                os.path.join(load_dir, "models", self.name, "critic")
            )
        else:
            self.actor.load_checkpoint()
            self.critic.load_checkpoint()
        self.actor.train()
