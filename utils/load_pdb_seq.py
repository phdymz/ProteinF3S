import numpy as np
import torch
import os
from torchdrug import data
from torchdrug.data import Molecule, PackedMolecule, Dictionary, feature
from rdkit import Chem
import warnings
from torchdrug.core import Registry as R
from torchdrug import utils



class Protein_New_Seq(data.protein.Protein):


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





