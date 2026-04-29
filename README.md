# 13C NMR Chemical Shift Prediction

Ablation study comparing three feature representations (`topo`, `xyz`, `distances`) for per-atom 13C NMR chemical shift prediction using a graph attention network on NMRShiftDB2.

# IMPORTANT NOTE
All the ipynb resulting graphs all exist in the project. If you would like to run it without training the model, please use the attached JSONs, which contain all the data during training. Ensure raw data in data/raw.

Keep in mind, these JSONs will be overwritten if you train a model.

---

## Setup

```bash
conda activate CNMR_Shift_GNN
# or prefix any command below with: conda run -n CNMR_Shift_GNN
```

Place the raw dataset at `data/raw/nmrshiftdb2withsignals.sd` before running anything.

---

## Dataset setup

The raw SDF is not included in this repo. Download `nmrshiftdb2withsignals.sd` from [NMRShiftDB2](https://nmrshiftdb.nmr.uni-koeln.de/) and place it as follows:

```
mkdir -p data/raw
mv nmrshiftdb2withsignals.sd data/raw/
```

The `data/` directory is gitignored, so nothing in there will be committed.

---

## Commands

**Process the dataset** (run once — generates graph files in `data/processed/`)
```bash
python src/data_processer/dataset.py           # full dataset
python src/data_processer/dataset.py 5000      # mini subset of 5000 molecules
```

**Verify splits and dataloaders**
```bash
python src/data_processer/dataloader.py
```

**Train** (set `MODE`, `EPOCHS`, `LR`, etc. at the top of the file first)
```bash
python src/train/train.py
```

**Hyperparameter tuning** (Bayesian search via Optuna)
```bash
python src/train/tune.py
python src/train/tune.py --mode topo --n_trials 30 --k_mols 2000
```

Searches over `lr` (1e-4 to 1e-2, log scale), `hidden_dim` (64, 128, or 256), and `dropout` (0.0 to 0.3). `num_layers` is fixed at 3 and each trial trains for 20 epochs. `--k_mols` controls how many molecules are randomly sampled for tuning — smaller is faster but noisier.

**Evaluate a saved checkpoint**
```bash
python src/test/test_model.py
python src/test/test_model.py --mode topo
```

---

## Output files

**`best_model_{mode}.pth`** — saved during training whenever validation MAE improves. Reloaded automatically at the end of `train.py` for the final test evaluation. Gitignored.

**`results_{mode}.json`** — written by `train.py` after each full run.
```json
{
  "config":       { "mode", "epochs", "lr", "hidden_dim", "num_layers", "dropout" },
  "train_maes":   [],   // MAE per epoch
  "val_maes":     [],   // MAE per epoch
  "best_val_mae": 0.0,
  "test_mae":     0.0,
  "test_pred":    [],   // predicted shifts for all labelled carbons in the test set
  "test_true":    []    // ground-truth shifts for the same atoms
}
```

```python
import json
r = json.load(open("results_distances.json"))
```

**`BO_results_{mode}.json`** — written by `tune.py`. Contains the best hyperparameters found and the full trial history.

---

## Project structure

```
src/
  data_processer/
    dataset.py      # SDF → PyG graph objects
    dataloader.py   # train/val/test splits
  models/
    model.py        # GAT model (topo / xyz / distances modes)
  train/
    train.py        # training loop
    tune.py         # Bayesian hyperparameter search
  test/
    test_model.py   # evaluate a saved checkpoint
data/
  raw/              # nmrshiftdb2withsignals.sd goes here
  processed/        # auto-generated, gitignored
```
