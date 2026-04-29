# GATConv-based node regression model for 13C NMR chemical shift prediction.
#
# The mode argument controls how much geometric information the model sees:
#   'topo'      -- pure graph topology, no geometry at all
#   'xyz'       -- 3D coordinates appended to each node's feature vector
#   'distances' -- bond length included as an edge feature (the default)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

_VALID_MODES = ("topo", "xyz", "distances")


class NMRTopological2D(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        mode: str = "distances",
    ):
        super().__init__()

        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
        self.mode = mode
        self.dropout = dropout

        # xyz mode pads node features with x/y/z coords; distances mode keeps full edge_attr
        node_in = num_node_features + (3 if mode == "xyz" else 0)
        edge_in = num_edge_features if mode == "distances" else num_edge_features - 1

        self.node_encoder = nn.Sequential(
            nn.Linear(node_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, edge_dim=edge_in, concat=False)
            for _ in range(num_layers)
        ])
        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr, pos=None):
        if self.mode == "xyz":
            x = torch.cat([x, pos], dim=-1)
        if self.mode != "distances":
            edge_attr = edge_attr[:, :7]

        x = self.node_encoder(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = F.gelu(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
            # no dropout after the last layer

        return self.output_head(x)  # shape [N, 1], one predicted shift per atom


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    for mode in _VALID_MODES:
        model = NMRTopological2D(
            num_node_features=70,
            num_edge_features=8,
            mode=mode,
        ).to(device)
        print(f"\nmode={mode!r}")
        print(model)
