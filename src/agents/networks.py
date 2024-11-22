import os

import torch as T
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv, GCNConv


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, lr, chkpt_dir, name):
        super().__init__()
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
        self.chkpt = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc1(x)


class MLP(nn.Module):
    """
    Simple 2 layer MLP
    """

    def __init__(
        self,
        in_dim,
        h_dim,
        out_dim,
        lr,
        chkpt_dir,
        name,
        final_activation=None,
    ):
        super().__init__()
        self.chkpt = os.path.join(chkpt_dir, name)
        self.device = T.device(
            # "cuda:0" if T.cuda.is_available() else "cpu"
            "cpu"
        )
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.ln1 = nn.LayerNorm(h_dim)
        self.fc2 = nn.Linear(h_dim, out_dim)
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)
        self.final_activation = final_activation

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.ln1(x)
        x = self.fc2(x)

        if self.final_activation == "sigmoid":
            x = F.sigmoid(x)
        elif self.final_activation == "softmax":
            x = F.softmax(x)

        return x

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt)

    def load_checkpoint(self, chkpt=None):
        if chkpt is not None:
            self.load_state_dict(T.load(chkpt))
        else:
            self.load_state_dict(T.load(self.chkpt))


class MLP2(nn.Module):
    """
    Simple 3 layer MLP
    """

    def __init__(
        self,
        in_dim,
        h_dim,
        out_dim,
        lr,
        chkpt_dir,
        name,
        final_activation=None,
    ):
        super().__init__()
        self.chkpt = os.path.join(chkpt_dir, name)
        self.device = T.device(
            # "cuda:0" if T.cuda.is_available() else "cpu"
            "cpu"
        )

        self.fc1 = nn.Linear(in_dim, h_dim)
        self.ln = nn.LayerNorm(h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, out_dim)
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)
        self.final_activation = final_activation

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.ln(x)
        x = F.relu(self.fc2(x))
        x = self.ln(x)
        x = self.fc3(x)

        if self.final_activation == "sigmoid":
            x = F.sigmoid(x)
        elif self.final_activation == "softmax":
            x = F.softmax(x)

        return x

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt)

    def load_checkpoint(self, chkpt=None):
        if chkpt is not None:
            self.load_state_dict(T.load(chkpt))
        else:
            self.load_state_dict(T.load(self.chkpt))


class ActorNetwork(nn.Module):
    def __init__(
        self,
        in_dim,
        h_dim,
        chkpt_dir,
        lr=1e-4,
        softmax_temp=1,
        name="actor",
        device="cpu",
    ):
        super().__init__()
        self.chkpt = os.path.join(chkpt_dir, name)
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.ln1 = nn.LayerNorm(h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.ln2 = nn.LayerNorm(h_dim)
        self.fc3 = nn.Linear(h_dim, 1)
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
        self.softmax_temp = softmax_temp
        self.device = T.device(device)
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.ln1(x)
        x = F.relu(self.fc2(x))
        x = self.ln2(x)
        x = self.fc3(x)
        x = F.softmax(x.squeeze(-1) / self.softmax_temp, dim=-1)

        return x

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt)

    def load_checkpoint(self, chkpt=None):
        if chkpt is not None:
            self.load_state_dict(
                T.load(chkpt, map_location=self.device)
            )
        else:
            self.load_state_dict(
                T.load(self.chkpt, map_location=self.device)
            )


class CriticNetwork(nn.Module):
    def __init__(
        self,
        in_dim,
        h_dim,
        chkpt_dir,
        name="critic",
        lr=1e-4,
        device="cpu",
    ):
        super().__init__()
        self.chkpt = os.path.join(chkpt_dir, name)
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.ln1 = nn.LayerNorm(h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.ln2 = nn.LayerNorm(h_dim)
        self.fc3 = nn.Linear(h_dim, 1)
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
        self.device = T.device(device)
        self.to(device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.ln1(x)
        x = F.relu(self.fc2(x))
        x = self.ln2(x)
        x = self.fc3(x)

        x = F.sigmoid(x)
        return x

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt)

    def load_checkpoint(self, chkpt=None):
        if chkpt is not None:
            self.load_state_dict(
                T.load(chkpt, map_location=self.device)
            )
        else:
            self.load_state_dict(
                T.load(self.chkpt, map_location=self.device)
            )


class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1):
        super(GATLayer, self).__init__()
        self.gat_conv = GATv2Conv(
            in_channels, out_channels, heads=heads, concat=True
        )
        self.layer_norm = nn.LayerNorm(out_channels * heads)

    def forward(self, x, edge_index):
        x = self.gat_conv(x, edge_index)
        x = self.layer_norm(x)
        return x


class GAT(nn.Module):
    def __init__(
        self,
        in_channels,
        h_channels,
        out_channels,
        chkpt_dir,
        lr=1e-4,
        layers=3,
        heads=1,
        name="gnn",
        device="cpu",
    ):
        super(GAT, self).__init__()
        self.conv1 = GATv2Conv(in_channels, h_channels, heads=heads)
        self.conv2 = GATv2Conv(
            h_channels * heads, out_channels, heads=heads
        )
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
        self.device = T.device(device)
        self.chkpt = os.path.join(chkpt_dir, name)

        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(
            GATLayer(in_channels, h_channels, heads=heads)
        )

        # Hidden layers
        for _ in range(layers - 2):
            self.layers.append(
                GATLayer(heads * h_channels, h_channels, heads=heads)
            )

        # Output layer
        self.layers.append(
            GATLayer(heads * h_channels, out_channels, heads=1)
        )

        self.to(device)

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
        return x

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt)

    def load_checkpoint(self, chkpt=None):
        if chkpt is not None:
            self.load_state_dict(
                T.load(chkpt, map_location=self.device)
            )
        else:
            self.load_state_dict(
                T.load(self.chkpt, map_location=self.device)
            )


class GCN(nn.Module):
    def __init__(
        self,
        in_channels,
        h_channels,
        out_channels,
        chkpt_dir,
        lr=1e-3,
        device="cpu",
        name="gnn",
    ):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, h_channels)
        self.ln = nn.LayerNorm(h_channels)
        self.conv2 = GATv2Conv(h_channels, out_channels)
        self.device = T.device(device)
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)
        self.chkpt = os.path.join(chkpt_dir, name)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.ln(x)
        x = F.relu(self.conv2(x, edge_index))
        return x

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt)

    def load_checkpoint(self, chkpt=None):
        if chkpt is not None:
            self.load_state_dict(T.load(chkpt))
        else:
            self.load_state_dict(T.load(self.chkpt))
