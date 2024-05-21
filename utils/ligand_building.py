import numpy as np
import torch
import os
from tqdm import tqdm
from torchdrug import data
import utils.load_pdb
import json
from torch_geometric.data import Data
import networkx as nx
from typing import List, Union, Set, Any
from rdkit import Chem




#Kyte-Doolittle scale for hydrophobicity
KD_SCALE = {}
KD_SCALE["ILE"] = 4.5
KD_SCALE["VAL"] = 4.2
KD_SCALE["LEU"] = 3.8
KD_SCALE["PHE"] = 2.8
KD_SCALE["CYS"] = 2.5
KD_SCALE["MET"] = 1.9
KD_SCALE["ALA"] = 1.8
KD_SCALE["GLY"] = -0.4
KD_SCALE["THR"] = -0.7
KD_SCALE["SER"] = -0.8
KD_SCALE["TRP"] = -0.9
KD_SCALE["TYR"] = -1.3
KD_SCALE["PRO"] = -1.6
KD_SCALE["HIS"] = -3.2
KD_SCALE["GLU"] = -3.5
KD_SCALE["GLN"] = -3.5
KD_SCALE["ASP"] = -3.5
KD_SCALE["ASN"] = -3.5
KD_SCALE["LYS"] = -3.9
KD_SCALE["ARG"] = -4.5

# Symbols for different atoms
ATOM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', \
    'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', \
    'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', \
    'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', \
    'Ce','Gd','Ga','Cs', '*', 'unk']

AMINO_ACIDS = [
    'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET',
    'ASN', 'PYL', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'SEC', 'VAL', 'TRP', 'TYR',
    'unk'
]

SECONDARY_STRUCTS = ['H', 'G', 'I', 'E', 'B', 'T', 'C', 'unk']

MAX_NB = 10
DEGREES = list(range(MAX_NB))
EXP_VALENCE = [1, 2, 3, 4, 5, 6]
IMP_VALENCE = [0, 1, 2, 3, 4, 5]

BOND_TYPES = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, \
    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
BOND_FLOAT_TO_TYPE = {
    0.0: BOND_TYPES[0],
    1.0: BOND_TYPES[1],
    2.0: BOND_TYPES[2],
    3.0: BOND_TYPES[3],
    1.5: BOND_TYPES[4],
}

BOND_FLOATS = [0.0, 1.0, 2.0, 3.0, 1.5]

ATOM_FDIM = len(ATOM_LIST) + len(DEGREES) + len(EXP_VALENCE) + len(
    IMP_VALENCE) + 1
BOND_FDIM = 6
CONTACT_FDIM = 2
SURFACE_NODE_FDIM = 4
SURFACE_EDGE_FDIM = 5
PATCH_NODE_FDIM = 4
PATCH_EDGE_FDIM = 4




class AtomProp(object):
    """Wrapper class that holds all properties of an atom."""

    def __init__(self, atom: Chem.Atom) -> None:
        """
        Parameters
        ----------
        atom: Chem.Atom,
            Instance of rdkit.Chem.Atom
        """
        self.symbol = atom.GetSymbol()
        self.degree = atom.GetDegree()
        self.exp_valence = atom.GetExplicitValence()
        self.imp_valence = atom.GetImplicitValence()
        self.is_aromatic = atom.GetIsAromatic()


def onek_encoding_unk(x: Any, allowable_set: Union[List, Set]) -> List:
    """Converts x to one hot encoding.

    Parameters
    ----------
    x: Any,
        An element of any type
    allowable_set: Union[List, Set]
        Allowable element collection

    Returns
    -------
    list, indicating the one hot encoding of x in allowable_set
    """
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: float(x == s), allowable_set))



def get_atom_features(atom_prop: AtomProp, **kwargs) -> np.ndarray:
    """
    Get atom features. The atom features computed

    Parameters
    ----------
    atom: Chem.Atom,
        Atom object from RDKit

    Returns
    -------
    atom_features: np.ndarray,
        Array of atom features
    """
    if atom_prop == "*":
        return np.array([0] * ATOM_FDIM)
    atom_features = np.array(
        onek_encoding_unk(atom_prop.symbol, ATOM_LIST) +
        onek_encoding_unk(atom_prop.degree, DEGREES) +
        onek_encoding_unk(atom_prop.exp_valence, EXP_VALENCE) +
        onek_encoding_unk(atom_prop.imp_valence, IMP_VALENCE) +
        [float(atom_prop.is_aromatic)])
    return atom_features


class BondProp(object):
    """Wrapper class that holds all properties of a bond."""

    def __init__(self, bond: Chem.Bond) -> None:
        """
        Parameters
        ----------
        bond: Chem.Bond,
            Instance of rdkit.Chem.Bond
        """
        self.bond_type = bond.GetBondType()
        self.is_conj = bond.GetIsConjugated()
        self.is_ring = bond.IsInRing()


def get_bond_features(bond_prop: BondProp, **kwargs) -> np.ndarray:
    """
    Get bond features. Features computed are a one hot encoding of the bond type,
    its aromaticity and ring membership.

    Parameters
    ----------
    bond: Chem.Bond,
        bond object

    Returns
    -------
    bond_features: np.ndarray,
        Array of bond features
    """
    if bond_prop == "*":
        return np.array([0] * BOND_FDIM)
    bt = bond_prop.bond_type
    bond_features = [float(bt == bond_type) for bond_type in BOND_TYPES[1:]]
    bond_features.extend([float(bond_prop.is_conj), float(bond_prop.is_ring)])
    bond_features = np.array(bond_features, dtype=np.float32)
    return bond_features


def create_pad_tensor(alist):
    max_len = max([len(a) for a in alist]) + 1
    for a in alist:
        pad_len = max_len - len(a)
        a.extend([0] * pad_len)
    return torch.tensor(alist, dtype=torch.long)



class LigandMol:

    def build(self, lig_smi: str, target = None) -> Data:
        mol = Chem.MolFromSmiles(lig_smi)
        if mol is None:
            print(f"Mol is None {lig_smi}", flush=True)
            return None

        data = {}
        G = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(mol))
        G = nx.convert_node_labels_to_integers(G)
        edge_index = torch.tensor(list(G.edges)).t().contiguous()

        x = []
        for atom in mol.GetAtoms():
            x.append(get_atom_features(AtomProp(atom)))
        data['x'] = torch.tensor(x).float()

        edge_attr = []
        mess_idx, edge_dict = [[]], {}

        for a1, a2 in G.edges():
            bond = mol.GetBondBetweenAtoms(a1, a2)
            bond_feat = get_bond_features(BondProp(bond))
            edge_attr.append(bond_feat)
            edge_dict[(a1, a2)] = eid = len(edge_dict) + 1
            mess_idx.append([])

        data['edge_attr'] = torch.tensor(edge_attr).float()
        data['edge_index'] = edge_index

        for u, v in G.edges:
            eid = edge_dict[(u, v)]
            for w in G.predecessors(u):
                if w == v: continue
                mess_idx[eid].append(edge_dict[(w, u)])
        mess_idx = create_pad_tensor(mess_idx)
        data['mess_idx'] = mess_idx

        def finite_check(x):
            return torch.isfinite(x).all().item()

        data = Data.from_dict(data)
        checks = [finite_check(data.x), finite_check(data.edge_attr)]
        if target is not None:
            target = torch.tensor(target)
            if not len(target.shape):
                target = target.unsqueeze(0)
            data.y = target
            checks += [finite_check(data.y)]

        if not all(checks):
            print(f"Nan checks failed for ligand: {lig_smi}", flush=True)
            return None

        return data



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Process_ligand')
    parser.add_argument('--dataset', default='pdbbind', type=str,
                        choices=['func', 'ec', 'pdbbind', 'fold'])
    parser.add_argument('--data_dir', default='/media/ymz/807a6847-34c1-4630-bb53-2a277f79be0c/ProteinF3S/pdbbind', type=str,
                        metavar='N', help='data root directory')
    parser.add_argument('--pdb_fix', default=False, type=bool)
    parser.add_argument('--surface', default=False, type=bool)
    parser.add_argument('--sequence', default=False, type=bool)

    args = parser.parse_args()
    return args


def main(args, split):

    if args.dataset == 'pdbbind':

        root = args.data_dir
        output_files = os.path.join(root, 'ligand')

        with open(os.path.join(root, "metadata/affinities.json"), "r") as f:
            affinity_dict = json.load(f)
        with open(os.path.join(root, "metadata/lig_smiles.json"), "r") as f:
            lig_smiles = json.load(f)
        protein_names = list(affinity_dict.keys())

        mol_builder = LigandMol()

        for protein_name in tqdm(protein_names):
            lig_smi = lig_smiles[protein_name + '_ligand']
            ligand = mol_builder.build(lig_smi)
            activity = affinity_dict[protein_name]

            output_file = os.path.join(output_files, protein_name)
            y = torch.tensor(activity)
            if not len(y.shape):
                y = y.unsqueeze(0)
            ligand.y = y

            processed_files = os.path.join(output_file, protein_name + '.pth')
            os.makedirs(output_file, exist_ok=True)

            torch.save(ligand, processed_files)

    print('processed ' + split + ' set')


if __name__ == '__main__':
    args = parse_args()
    for split in ['training', 'validation', 'testing']:
        main(args, split)

    print('Finish')


