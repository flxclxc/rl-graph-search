import os

import networkx as nx
import torch as T
import torch_geometric as tg
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from agents.a2c import A2C
from agents.networks import GAT, ActorNetwork, CriticNetwork


class GNNA2C(A2C):
    def __init__(
        self,
        env,
        config,
        chkpt_dir,
        name="gnn_a2c",
        device="cpu",
        use_degree=True,
        use_pca=False,
        nodes=None,
        ego_graphs=None,
        use_entropy=True,
    ):
        super().__init__(
            env,
            config,
            use_ego_graph=True,
            nodes=nodes,
            ego_graphs=ego_graphs,
            use_entropy=use_entropy,
            use_pca=use_pca,
        )

        self.name = name
        self.save_dir = os.path.join(chkpt_dir, name)

        self.attribute_dim = self.ego_graphs[0].x.shape[1] - 1
        self.rep_dim = config["rep_dim"]

        self.degrees = T.tensor(
            list(nx.get_node_attributes(env.g, "degree").values()),
            dtype=T.float32,
        )
        # Networks
        self.actor = ActorNetwork(
            in_dim=2 * self.rep_dim + 1,
            h_dim=config["h_actor"],
            chkpt_dir=os.path.join(chkpt_dir, name),
            name="actor",
            device=device,
        )

        self.critic = CriticNetwork(
            in_dim=2 * self.rep_dim + 1,
            h_dim=config["h_critic"],
            chkpt_dir=os.path.join(chkpt_dir, name),
            name="critic",
            device=device,
        )

        self.gnn = GAT(
            in_channels=self.attribute_dim + 1,
            h_channels=config["h_rep"],
            out_channels=self.rep_dim,
            layers=config["rep_layers"],
            chkpt_dir=os.path.join(chkpt_dir, name),
            name="gnn",
            device=device,
        )

        parameters = (
            list(self.actor.parameters())
            + list(self.critic.parameters())
            + list(self.gnn.parameters())
        )

        self.optimizer = T.optim.Adam(parameters, lr=config["lr"])
        self.node_embeddings = None

        self.loader = DataLoader(
            self.ego_graphs, batch_size=config["batch_size"]
        )
        self.batch = Batch.from_data_list(self.ego_graphs).batch.to(
            self.gnn.device
        )

        self.train_mode = True
        self.attributes = None

    def embed_ego_graphs(self):
        """Create vector representations for each ego graph observation."""
        graph_embs = []

        for batch_data in self.loader:
            batch_data = batch_data.to(self.gnn.device)
            output = self.gnn(batch_data.x, batch_data.edge_index)
            graph_embs.append(output)
            del output, batch_data

        # Convert the list of outputs to a single tensor
        all_outputs_tensor = T.cat(graph_embs, dim=0)
        del graph_embs

        # Calculate the mean over the node features for each graph
        attributes = tg.utils.scatter(
            all_outputs_tensor, self.batch, dim=0, reduce="mean"
        ).to(self.actor.device)
        del all_outputs_tensor

        attributes = T.cat(
            (
                attributes,
                self.degrees.unsqueeze(-1),
            ),
            dim=-1,
        )

        self.attributes = attributes

    def set_target(self, target, message):
        if self.train_mode or self.attributes is None:
            self.embed_ego_graphs()

        self.message = self.attributes[target][:-1].to(self.actor.device)
        self.message

    def save_checkpoints(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.gnn.save_checkpoint()

        T.save(
            self.optimizer.state_dict(),
            os.path.join(self.save_dir, "optimizer.pt"),
        )

    def load_checkpoints(self, load_dir=None):
        if load_dir is not None:
            self.actor.load_checkpoint(
                os.path.join(load_dir, "models", self.name, "actor")
            )
            self.critic.load_checkpoint(
                os.path.join(load_dir, "models", self.name, "critic")
            )
            self.gnn.load_checkpoint(
                os.path.join(load_dir, "models", self.name, "gnn")
            )

            if os.path.exists(
                os.path.join(load_dir, "models", self.name, "optimizer.pt")
            ):
                self.optimizer.load_state_dict(
                    T.load(
                        os.path.join(
                            load_dir,
                            "models",
                            self.name,
                            "optimizer.pt",
                        )
                    )
                )

        else:
            self.actor.load_checkpoint()
            self.critic.load_checkpoint()
            self.gnn.load_checkpoint()
            if os.path.exists(os.path.join(self.save_dir, "optimizer.pt")):
                self.optimizer.load_state_dict(
                    T.load(os.path.join(self.save_dir, "optimizer.pt"))
                )
