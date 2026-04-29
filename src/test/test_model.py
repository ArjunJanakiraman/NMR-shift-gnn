# Loads a saved checkpoint and evaluates it on the test split.
#
# Usage:
#   conda run -n CNMR_Shift_GNN python src/test/test_model.py
#   conda run -n CNMR_Shift_GNN python src/test/test_model.py --mode topo
#   conda run -n CNMR_Shift_GNN python src/test/test_model.py --mode xyz

import sys
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse

import torch
import torch.nn.functional as F

from data_processer.dataloader import get_dataloaders
from models.model import NMRTopological2D

# defaults — override via CLI flags
MODE       = "distances"   # "topo" | "xyz" | "distances"
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT    = 0.1
CHECKPOINT_DIR = "."  # directory where best_model_{mode}.pth lives


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=MODE, choices=["topo", "xyz", "distances"])
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device} | Mode: {args.mode}")

    loaders, dataset = get_dataloaders()
    test_loader = loaders["test"]

    sample = dataset[0]
    num_node_features = sample.x.shape[1]
    num_edge_features = sample.edge_attr.shape[1]

    model = NMRTopological2D(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        mode=args.mode,
    ).to(device)

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_model_{args.mode}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    total_loss = 0.0
    total_atoms = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.pos)
            # evaluate only on labelled carbons
            loss = F.l1_loss(out[batch.mask], batch.y[batch.mask])
            n = batch.mask.sum().item()
            total_loss += loss.item() * n
            total_atoms += n

    test_mae = total_loss / total_atoms
    print(f"Test MAE ({args.mode}): {test_mae:.4f} ppm")


if __name__ == "__main__":
    main()
