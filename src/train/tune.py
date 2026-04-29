# Bayesian hyperparameter search for NMRTopological2D using Optuna (TPE sampler).
# Tunes lr, hidden_dim, and dropout with num_layers and epochs held fixed.
# A random subset of k_mols molecules is sampled once at startup with a fixed seed,
# so re-running with the same --k_mols flag always uses the same molecules.
#
# Usage:
#   conda run -n CNMR_Shift_GNN python src/train/tune.py
#   conda run -n CNMR_Shift_GNN python src/train/tune.py --mode topo
#   conda run -n CNMR_Shift_GNN python src/train/tune.py --mode xyz --n_trials 30 --k_mols 2000

import sys
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import optuna
from optuna.samplers import TPESampler

from data_processer.dataset import NMRShiftDataset
from models.model import NMRTopological2D

# fixed settings shared across all trials
TUNE_EPOCHS = 20
NUM_LAYERS  = 3
BATCH_SIZE  = 128
K_MOLS      = 3000   # default subset size; override with --k_mols
SAMPLE_SEED = 42     # fixed seed for subset sampling so results are reproducible
TRAIN_RATIO = 0.8


BASE_SEED = 42


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


def build_loaders(k_mols: int, data_root: str = "data/"):
    """Loads the full dataset, draws k_mols molecules with a fixed seed, and returns 80/20 loaders."""
    dataset = NMRShiftDataset(root=data_root)
    n = len(dataset)
    k = min(k_mols, n)

    rng = torch.Generator()
    rng.manual_seed(SAMPLE_SEED)
    perm = torch.randperm(n, generator=rng)[:k]
    k_dataset = dataset[perm.tolist()]

    n_train = int(k * TRAIN_RATIO)
    train_loader = DataLoader(k_dataset[:n_train], batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(k_dataset[n_train:],  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    sample = k_dataset[0]
    print(f"Sampled {k}/{n} molecules | train={n_train} val={k - n_train}")
    return {"train": train_loader, "val": val_loader}, sample


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_atoms = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.pos)
        loss = F.l1_loss(out[batch.mask], batch.y[batch.mask])
        loss.backward()
        optimizer.step()
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


def make_objective(loaders, sample, device, mode):
    num_node_features = sample.x.shape[1]
    num_edge_features = sample.edge_attr.shape[1]

    def objective(trial):
        set_seed(BASE_SEED + trial.number)
        lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        dropout    = trial.suggest_float("dropout", 0.0, 0.3)

        model = NMRTopological2D(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_dim=hidden_dim,
            num_layers=NUM_LAYERS,
            dropout=dropout,
            mode=mode,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, TUNE_EPOCHS + 1):
            train_one_epoch(model, loaders["train"], optimizer, device)
            val_mae = evaluate(model, loaders["val"], device)
            trial.report(val_mae, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return val_mae

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",     default="distances", choices=["topo", "xyz", "distances"])
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--k_mols",   type=int, default=K_MOLS,
                        help="Number of molecules randomly sampled for tuning (reproducible via fixed seed)")
    args = parser.parse_args()

    set_seed(BASE_SEED)
    device = get_device()
    print(f"Device: {device} | Mode: {args.mode} | Trials: {args.n_trials} | k_mols: {args.k_mols}")

    loaders, sample = build_loaders(args.k_mols)

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(
        make_objective(loaders, sample, device, args.mode),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    print("\n=== Best trial ===")
    best = study.best_trial
    print(f"  Val MAE: {best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # collect per-trial results so we can inspect the full search history later
    trials_data = []
    for t in study.trials:
        trials_data.append({
            "trial":      t.number,
            "state":      t.state.name,   # COMPLETE, PRUNED, or FAIL
            "val_mae":    t.value if t.value is not None else None,
            "params":     t.params,
            "pruned_step": t.last_step if t.state.name == "PRUNED" else None,
        })

    bo_results = {
        "config": {
            "mode":        args.mode,
            "n_trials":    args.n_trials,
            "k_mols":      args.k_mols,
            "tune_epochs": TUNE_EPOCHS,
            "num_layers":  NUM_LAYERS,
            "batch_size":  BATCH_SIZE,
            "sample_seed": SAMPLE_SEED,
        },
        "best": {
            "val_mae":    best.value,
            "lr":         best.params["lr"],
            "hidden_dim": best.params["hidden_dim"],
            "dropout":    best.params["dropout"],
        },
        "trials": trials_data,
    }

    out_path = f"BO_results_{args.mode}.json"
    with open(out_path, "w") as f:
        json.dump(bo_results, f, indent=2)
    print(f"\nBO results saved to {out_path}")


if __name__ == "__main__":
    main()
