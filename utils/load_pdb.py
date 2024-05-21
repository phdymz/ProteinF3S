import numpy as np
import torch
import os
from torchdrug import data
from torchdrug.data import Molecule, PackedMolecule, Dictionary, feature
from rdkit import Chem
import warnings
from torchdrug.core import Registry as R
from torchdrug import utils



class Protein_New(data.protein.Protein):

    # add unknown amino category
    residue2id = {"GLY": 0, "ALA": 1, "SER": 2, "PRO": 3, "VAL": 4, "THR": 5, "CYS": 6, "ILE": 7, "LEU": 8,
                  "ASN": 9, "ASP": 10, "GLN": 11, "LYS": 12, "GLU": 13, "MET": 14, "HIS": 15, "PHE": 16,
                  "ARG": 17, "TYR": 18, "TRP": 19, "UNK": 20}


    id2residue = {v: k for k, v in residue2id.items()}
    id2residue_symbol = {0: "G", 1: "A", 2: "S", 3: "P", 4: "V", 5: "T", 6: "C", 7: "I", 8: "L", 9: "N",
                         10: "D", 11: "Q", 12: "K", 13: "E", 14: "M", 15: "H", 16: "F", 17: "R", 18: "Y", 19: "W", 20: "X"}
    residue_symbol2id = {v: k for k, v in id2residue_symbol.items()}

    @classmethod
    @utils.deprecated_alias(node_feature="atom_feature", edge_feature="bond_feature", graph_feature="mol_feature")
    def from_molecule(cls, mol, atom_feature="default", bond_feature="default", residue_feature="default",
                      mol_feature=None, kekulize=False):
        """
        Create a protein from an RDKit object.

        Parameters:
            mol (rdchem.Mol): molecule
            atom_feature (str or list of str, optional): atom features to extract
            bond_feature (str or list of str, optional): bond features to extract
            residue_feature (str, list of str, optional): residue features to extract
            mol_feature (str or list of str, optional): molecule features to extract
            kekulize (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edge_list``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        protein = Molecule.from_molecule(mol, atom_feature=atom_feature, bond_feature=bond_feature,
                                         mol_feature=mol_feature, with_hydrogen=False, kekulize=kekulize)
        residue_feature = cls._standarize_option(residue_feature)

        if kekulize:
            Chem.Kekulize(mol)

        residue_type = []
        atom_name = []
        is_hetero_atom = []
        occupancy = []
        b_factor = []
        atom2residue = []
        residue_number = []
        insertion_code = []
        chain_id = []
        _residue_feature = []
        last_residue = None
        atoms = [mol.GetAtomWithIdx(i) for i in range(mol.GetNumAtoms())] + [cls.dummy_atom]
        for atom in atoms:
            pdbinfo = atom.GetPDBResidueInfo()
            number = pdbinfo.GetResidueNumber()
            code = pdbinfo.GetInsertionCode()
            type = pdbinfo.GetResidueName().strip()
            canonical_residue = (number, code, type)
            if canonical_residue != last_residue:
                last_residue = canonical_residue
                if type not in cls.residue2id:
                    warnings.warn("Unknown residue `%s`." % type)
                    type = "UNK"
                residue_type.append(cls.residue2id[type])
                residue_number.append(number)
                if pdbinfo.GetInsertionCode() not in cls.alphabet2id or pdbinfo.GetChainId() not in cls.alphabet2id:
                    return None
                insertion_code.append(cls.alphabet2id[pdbinfo.GetInsertionCode()])
                chain_id.append(cls.alphabet2id[pdbinfo.GetChainId()])
                feature = []
                for name in residue_feature:
                    func = R.get("features.residue.%s" % name)
                    feature += func(pdbinfo)
                _residue_feature.append(feature)
            name = pdbinfo.GetName().strip()
            if name not in cls.atom_name2id:
                name = "UNK"
            atom_name.append(cls.atom_name2id[name])
            is_hetero_atom.append(pdbinfo.GetIsHeteroAtom())
            occupancy.append(pdbinfo.GetOccupancy())
            b_factor.append(pdbinfo.GetTempFactor())
            atom2residue.append(len(residue_type) - 1)
        residue_type = torch.tensor(residue_type)[:-1]
        atom_name = torch.tensor(atom_name)[:-1]
        is_hetero_atom = torch.tensor(is_hetero_atom)[:-1]
        occupancy = torch.tensor(occupancy)[:-1]
        b_factor = torch.tensor(b_factor)[:-1]
        atom2residue = torch.tensor(atom2residue)[:-1]
        residue_number = torch.tensor(residue_number)[:-1]
        insertion_code = torch.tensor(insertion_code)[:-1]
        chain_id = torch.tensor(chain_id)[:-1]
        if len(residue_feature) > 0:
            _residue_feature = torch.tensor(_residue_feature)[:-1]
        else:
            _residue_feature = None

        return cls(protein.edge_list, num_node=protein.num_node, residue_type=residue_type,
                   atom_name=atom_name, atom2residue=atom2residue, residue_feature=_residue_feature,
                   is_hetero_atom=is_hetero_atom, occupancy=occupancy, b_factor=b_factor,
                   residue_number=residue_number, insertion_code=insertion_code, chain_id=chain_id,
                   meta_dict=protein.meta_dict, **protein.data_dict)

    @classmethod
    @utils.deprecated_alias(node_feature="atom_feature", edge_feature="bond_feature", graph_feature="mol_feature")
    def from_pdb(cls, pdb_file, atom_feature="default", bond_feature="default", residue_feature="default",
                 mol_feature=None, kekulize=False):
        """
        Create a protein from a PDB file.

        Parameters:
            pdb_file (str): file name
            atom_feature (str or list of str, optional): atom features to extract
            bond_feature (str or list of str, optional): bond features to extract
            residue_feature (str, list of str, optional): residue features to extract
            mol_feature (str or list of str, optional): molecule features to extract
            kekulize (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edge_list``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        if not os.path.exists(pdb_file):
            raise FileNotFoundError("No such file `%s`" % pdb_file)
        mol = Chem.MolFromPDBFile(pdb_file, sanitize = False)
        if mol is None:
            raise ValueError("RDKit cannot read PDB file `%s`" % pdb_file)
        return cls.from_molecule(mol, atom_feature, bond_feature, residue_feature, mol_feature, kekulize)





