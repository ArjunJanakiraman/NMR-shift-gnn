# Dataset pipeline for 13C NMR chemical shift prediction.
# Builds atom/bond vocabularies from the raw SDF, featurizes molecules
# into PyG graphs with 3D conformers, and wraps everything in a Dataset class.

# --- Imports & constants ---

import os
from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch_geometric.data import Data, Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

USE_MMFF = True  # set to False to skip MMFF refinement and just use the raw ETKDGv3 conformer

DUMMY_SHIFT = -1000.0   # placeholder shift for atoms that have no ground-truth label
SHIFT_MIN = -50.0       # shifts outside this range are treated as outliers and dropped
SHIFT_MAX = 300.0
RING_SIZES = [3, 4, 5, 6, 7, 8]  # ring sizes we encode as binary features on each atom

BOND_VOCAB = [
    str(Chem.rdchem.BondType.SINGLE),
    str(Chem.rdchem.BondType.DOUBLE),
    str(Chem.rdchem.BondType.TRIPLE),
    str(Chem.rdchem.BondType.AROMATIC),
    "UNKNOWN",
]


# Vocabulary utilities 

def _collect_vocab_from_mol(mol_h, symbols, formal_charges, total_valences,
                             hybridizations, total_hs):
    """Adds atom property values from this molecule into the running vocab sets."""
    for atom in mol_h.GetAtoms():
        symbols.add(atom.GetSymbol())
        formal_charges.add(atom.GetFormalCharge())
        total_valences.add(atom.GetTotalValence())
        hybridizations.add(str(atom.GetHybridization()))
        total_hs.add(atom.GetTotalNumHs())


def build_vocabulary(sdf_path: str, vocab_path: str) -> Dict[str, List]:
    """
    Does one pass over the SDF to collect all unique atom property values.
    Each category is sorted and gets an UNKNOWN entry appended at the end
    so we can handle unseen values at inference time. Saves the result to disk.
    """
    symbols = set()
    formal_charges = set()
    total_valences = set()
    hybridizations = set()
    total_hs = set()

    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
    for mol in tqdm(supplier, desc="Vocab pass"):
        if mol is None:
            continue
        mol_h = Chem.AddHs(mol)
        try:
            Chem.SanitizeMol(mol_h)
        except Exception:
            continue
        _collect_vocab_from_mol(mol_h, symbols, formal_charges,
                                 total_valences, hybridizations, total_hs)

    # sort by (type name, value) so mixed-type sets (e.g. int and str) don't crash
    def sort_set(s):
        try:
            return sorted(s, key=lambda x: (type(x).__name__, x))
        except TypeError:
            return sorted(s, key=str)

    vocab = {
        "symbols": sort_set(symbols) + ["UNKNOWN"],
        "formal_charges": sort_set(formal_charges) + ["UNKNOWN"],
        "total_valences": sort_set(total_valences) + ["UNKNOWN"],
        "hybridizations": sort_set(hybridizations) + ["UNKNOWN"],
        "total_hs": sort_set(total_hs) + ["UNKNOWN"],
    }

    torch.save(vocab, vocab_path)
    return vocab


# 13C target parsing 

def parse_13c_property(prop_string: str) -> Dict[int, float]:
    """
    Parses the NMRShiftDB2 13C shift string into a dict of {atom_idx: shift}.
    The format is 'shift;;atom_idx|shift;;atom_idx|...'. Entries outside the
    allowed ppm range are dropped.
    """
    shift_map = {}
    entries = [e.strip() for e in prop_string.split("|") if e.strip()]
    for entry in entries:
        parts = entry.split(";")
        if len(parts) < 3:
            continue
        try:
            shift = float(parts[0])
            atom_idx = int(parts[2])
        except (ValueError, IndexError):
            continue
        if shift < SHIFT_MIN or shift > SHIFT_MAX:
            continue
        shift_map[atom_idx] = shift
    return shift_map


def extract_13c_shift_map(mol) -> Optional[Dict[int, float]]:
    """
    Looks for a 13C property in the molecule's metadata and returns the parsed
    shift map. Returns None if no 13C property is present.
    """
    props = mol.GetPropsAsDict()
    for key, value in props.items():
        if "13C" in key:
            return parse_13c_property(str(value))
    return None


# Encoding helpers 

def one_hot_encode(value, vocab: List) -> List[int]:
    """Standard one-hot encoding. If value isn't in the vocab, uses the UNKNOWN slot."""
    try:
        idx = vocab.index(value)
    except ValueError:
        idx = vocab.index("UNKNOWN")
    encoding = [0] * len(vocab)
    encoding[idx] = 1
    return encoding


# Feature builders 

def featurize_atom(atom, vocab: Dict[str, List], ring_info) -> List[int]:
    """
    Builds the node feature vector for a single atom. Concatenates one-hot
    encodings for symbol, formal charge, valence, hybridization, and H count,
    then appends binary ring-membership bits and an aromaticity flag.
    """
    features = []
    features += one_hot_encode(atom.GetSymbol(), vocab["symbols"])
    features += one_hot_encode(atom.GetFormalCharge(), vocab["formal_charges"])
    features += one_hot_encode(atom.GetTotalValence(), vocab["total_valences"])
    features += one_hot_encode(str(atom.GetHybridization()), vocab["hybridizations"])
    features += one_hot_encode(atom.GetTotalNumHs(), vocab["total_hs"])

    atom_idx = atom.GetIdx()
    for ring_size in RING_SIZES:
        features.append(int(ring_info.IsAtomInRingOfSize(atom_idx, ring_size)))

    features.append(int(atom.GetIsAromatic()))
    return features


def featurize_bond(bond, bond_length: float) -> List:
    """
    Builds the edge feature vector for a bond. Includes a bond-type one-hot
    (single/double/triple/aromatic/unknown), a conjugation flag, an in-ring flag,
    and the 3D bond length in angstroms from the MMFF conformer.
    """
    features = one_hot_encode(str(bond.GetBondType()), BOND_VOCAB)
    features.append(int(bond.GetIsConjugated()))
    features.append(int(bond.IsInRing()))
    features.append(bond_length)
    return features


# Graph assembly and conformer generation 

def generate_conformer(mol_h) -> Optional[Tensor]:
    """
    Generates a 3D conformer for the molecule using ETKDGv3, then refines it
    with MMFF if USE_MMFF is set. Returns an [N, 3] position tensor, or None
    if embedding fails.
    """
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol_h, params)
    if result == -1:
        return None
    if USE_MMFF:
        try:
            AllChem.MMFFOptimizeMolecule(mol_h)
        except Exception:
            return None

    conf = mol_h.GetConformer()
    pos = torch.tensor(
        [[conf.GetAtomPosition(i).x,
          conf.GetAtomPosition(i).y,
          conf.GetAtomPosition(i).z]
         for i in range(mol_h.GetNumAtoms())],
        dtype=torch.float
    )
    return pos


def build_graph(mol_h, vocab: Dict[str, List], shift_map: Dict[int, float],
                smiles: str, mol_id: int) -> Optional[Data]:
    """
    Assembles a PyG Data object from a hydrogen-explicit molecule. Generates
    the 3D conformer, featurizes nodes and edges, and fills in the shift targets
    and mask. Returns None if the conformer fails or there are no labelled shifts.
    """
    if not shift_map:
        return None

    pos = generate_conformer(mol_h)
    if pos is None:
        return None

    ring_info = mol_h.GetRingInfo()
    num_atoms = mol_h.GetNumAtoms()

    # build node feature matrix — one row per atom
    x = torch.tensor(
        [featurize_atom(atom, vocab, ring_info) for atom in mol_h.GetAtoms()],
        dtype=torch.float
    )

    # fill shift targets; only carbon atoms with a ground-truth label get a real value
    y = torch.full((num_atoms,), DUMMY_SHIFT, dtype=torch.float)
    mask = torch.zeros(num_atoms, dtype=torch.bool)
    for atom in mol_h.GetAtoms():
        if atom.GetAtomicNum() == 6:
            atom_idx = atom.GetIdx()
            if atom_idx in shift_map:
                y[atom_idx] = shift_map[atom_idx]
                mask[atom_idx] = True
    y = y.view(-1, 1)

    # build edge index and attributes; each bond is added in both directions
    edge_index_list = []
    edge_attr_list = []
    for bond in mol_h.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_length = (pos[i] - pos[j]).norm().item()
        feat = featurize_bond(bond, bond_length)
        edge_index_list.append([i, j])
        edge_attr_list.append(feat)
        edge_index_list.append([j, i])
        edge_attr_list.append(feat)

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 8), dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        y=y,
        mask=mask,
        smiles=smiles,
        mol_id=mol_id,
    )


# Dataset class, builds off pytorch geometric documentation

class NMRShiftDataset(Dataset):
    """PyG Dataset that wraps the NMRShiftDB2 SDF file for 13C shift prediction."""

    def __init__(self, root: str, transform=None, pre_transform=None,
                 pre_filter=None):
        self._root = root
        super().__init__(root, transform, pre_transform, pre_filter)

        metadata_path = os.path.join(self.processed_dir, "metadata.pt")
        metadata = torch.load(metadata_path, weights_only=True)
        self._len = metadata["length"]

    @property
    def processed_dir(self) -> str:
        return os.path.join(self._root, "processed")

    @property
    def raw_file_names(self) -> List[str]:
        return ["nmrshiftdb2withsignals.sd"]

    @property
    def processed_file_names(self) -> List[str]:
        return ["metadata.pt"]

    def download(self):
        pass  # the SDF file must be placed manually in raw_dir

    def process(self):
        # vocab is always stored in the main processed dir so mini-dataset runs can reuse it
        full_processed_dir = os.path.join(self._root, "processed")
        os.makedirs(full_processed_dir, exist_ok=True)
        vocab_path = os.path.join(full_processed_dir, "vocab.pt")
        sdf_path = os.path.join(self.raw_dir, self.raw_file_names[0])

        # only build vocab once; subsequent runs load from disk
        if not os.path.exists(vocab_path):
            build_vocabulary(sdf_path, vocab_path)

        vocab = torch.load(vocab_path, weights_only=True)

        supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
        data_idx = 0

        for mol in tqdm(supplier, desc="Processing pass"):
            if mol is None:
                continue

            shift_map = extract_13c_shift_map(mol)
            if shift_map is None:
                continue

            mol_h = Chem.AddHs(mol)
            try:
                Chem.SanitizeMol(mol_h)
            except Exception:
                continue

            Chem.FastFindRings(mol_h)

            # generate SMILES from a sanitized copy so it doesn't affect mol_h
            try:
                mol_clean = Chem.RWMol(mol)
                Chem.SanitizeMol(mol_clean)
                smiles = Chem.MolToSmiles(mol_clean)
            except Exception:
                smiles = ""

            mol_id = data_idx

            data = build_graph(mol_h, vocab, shift_map, smiles, mol_id)
            if data is None:
                continue

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            out_path = os.path.join(self.processed_dir, f"data_{data_idx}.pt")
            torch.save(data, out_path)
            data_idx += 1
            torch.save({"length": data_idx},
                       os.path.join(self.processed_dir, "metadata.pt"))

    def len(self) -> int:
        return self._len

    def get(self, idx: int) -> Data:
        path = os.path.join(self.processed_dir, f"data_{idx}.pt")
        return torch.load(path, weights_only=False)


if __name__ == "__main__":
    dataset = NMRShiftDataset(root="data/")
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample: {sample}")
