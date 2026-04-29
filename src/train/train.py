import sys
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import random
import numpy as np
import torch
import torch.nn.functional as F

from data_processer.dataloader import get_dataloaders
from models.model import NMRTopological2D

# training config — edit these before running
MODE       = "distances"   # "topo" | "xyz" | "distances"
EPOCHS     = 50
LR         = 0.001
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT    = 0.1


RANDOM_SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_atoms = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.pos)
        # only compute loss over labelled carbon atoms
        loss = F.l1_loss(out[batch.mask], batch.y[batch.mask])
        loss.backward()
        optimizer.step()

        # accumulate atom-weighted loss so we can report mean MAE at the end
        n = batch.mask.sum().item()
        total_loss += loss.item() * n
        total_atoms += n

    return total_loss / total_atoms


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_atoms = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.pos)
            loss = F.l1_loss(out[batch.mask], batch.y[batch.mask])

            n = batch.mask.sum().item()
            total_loss += loss.item() * n
            total_atoms += n

    return total_loss / total_atoms


def collect_predictions(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.pos)
            preds.extend(out[batch.mask].squeeze(-1).cpu().tolist())
            trues.extend(batch.y[batch.mask].squeeze(-1).cpu().tolist())
    return preds, trues


def main():
    set_seed(RANDOM_SEED)
    device = get_device()
    print(f"Using device: {device}")

    loaders, dataset = get_dataloaders()
    train_loader = loaders["train"]
    val_loader   = loaders["val"]
    test_loader  = loaders["test"]

    sample = dataset[0]
    num_node_features = sample.x.shape[1]
    num_edge_features = sample.edge_attr.shape[1]
    print(f"Node features: {num_node_features}, Edge features: {num_edge_features}")

    model = NMRTopological2D(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        mode=MODE,
    ).to(device)
    print(f"Model mode: {MODE}, Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_mae = float("inf")
    train_maes, val_maes = [], []

    for epoch in range(1, EPOCHS + 1):
        train_mae = train_one_epoch(model, train_loader, optimizer, device)
        val_mae   = evaluate(model, val_loader, device)

        train_maes.append(train_mae)
        val_maes.append(val_mae)

        print(f"Epoch {epoch:03d} | Train MAE: {train_mae:.3f} | Val MAE: {val_mae:.3f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), f"best_model_{MODE}.pth")
            print(f"  --> New best Val MAE: {best_val_mae:.3f}. Model saved.")

    # reload best checkpoint before running the final test evaluation
    model.load_state_dict(torch.load(f"best_model_{MODE}.pth", map_location=device))
    test_pred, test_true = collect_predictions(model, test_loader, device)
    test_mae = sum(abs(p - t) for p, t in zip(test_pred, test_true)) / len(test_pred)
    print(f"Test MAE: {test_mae:.3f}")

    results = {
        "config": {
            "mode": MODE,
            "epochs": EPOCHS,
            "lr": LR,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
        },
        "train_maes": train_maes,
        "val_maes": val_maes,
        "best_val_mae": best_val_mae,
        "test_mae": test_mae,
        "test_pred": test_pred,
        "test_true": test_true,
    }
    out_path = f"results_{MODE}.json"
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
