import torch
from torch_geometric.loader import DataLoader
from data_processer.dataset import NMRShiftDataset

DATA_ROOT = "data/"
BATCH_SIZE = 64
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
RANDOM_SEED = 42
NUM_WORKERS = 0


def get_dataloaders(data_root=DATA_ROOT, batch_size=BATCH_SIZE):
    # seed before shuffling so splits are reproducible across runs
    torch.manual_seed(RANDOM_SEED)
    dataset = NMRShiftDataset(root=data_root)
    dataset = dataset.shuffle()

    n = len(dataset)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    # slice the shuffled dataset into train / val / test
    train_dataset = dataset[:n_train]
    val_dataset = dataset[n_train:n_train + n_val]
    test_dataset = dataset[n_train + n_val:]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    return {"train": train_loader, "val": val_loader, "test": test_loader}, dataset


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    loaders, dataset = get_dataloaders()
    n = len(dataset)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val
    print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")
    batch = next(iter(loaders["train"]))
    print(batch)
