import numpy as np
import torch
import os
from torch.utils.data import Dataset
from tqdm import tqdm
from Bio.PDB import PDBParser, Polypeptide, MMCIFParser
# from torchdrug import data
import json


ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5, 'P': 6, 'NA': 7,
           'K': 8, 'MG': 9, 'CA': 10, 'FE': 11, 'ZN': 12, 'OTH': 13}


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Data_Pre_Process')
    parser.add_argument('--dataset', default='go', type=str,
                        choices=['func', 'ec', 'pdbbind', 'fold', 'go'])
    parser.add_argument('--data_dir', default='/media/ymz/807a6847-34c1-4630-bb53-2a277f79be0c/ProteinF3S/go', type=str,
                        metavar='N', help='data root directory')
    parser.add_argument('--pdb_fix', default=False, type=bool)
    parser.add_argument('--surface', default=True, type=bool)
    parser.add_argument('--sequence', default=False, type=bool)

    args = parser.parse_args()
    return args


def main(args, split):

    root = args.data_dir
    if args.pdb_fix:
        output_files = os.path.join(root, 'atom_fixed')
    else:
        output_files = os.path.join(root, 'atom')
    pdb_files = os.path.join(root, 'pdb_files')

    if args.dataset == 'func':
        fasta_file = os.path.join(root, 'chain_' + split + '.fasta')

        with open(fasta_file, 'r') as f:
            protein_names = []
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                    protein_names.append(protein_name.replace('.', '_'))
                else:
                    pass

        for protein_name in tqdm(protein_names):
            pdb_file = os.path.join(pdb_files, protein_name)
            if args.pdb_fix:
                pdb_root = os.path.join(pdb_file, protein_name + '_fixed.pdb')
                if not os.path.exists(pdb_root):
                    pdb_root = os.path.join(pdb_file, protein_name + '.pdb')
            else:
                pdb_root = os.path.join(pdb_file, protein_name + '.pdb')

            output_file = os.path.join(output_files, protein_name)
            os.makedirs(output_file, exist_ok=True)
            processed_file = os.path.join(output_file, protein_name + '.npz')

            if args.surface:
                parser = PDBParser()
                structure = parser.get_structure("structure", pdb_root)
                atoms = structure.get_atoms()

                pdb_atoms = []
                pdb_types = []

                for atom in atoms:
                    pdb_atoms.append(atom.get_coord())
                    if atom.element in ele2num:
                        pdb_types.append(ele2num[atom.element])
                    else:
                        pdb_types.append(ele2num['OTH'])

                coords = np.stack(pdb_atoms)
                types_array = np.zeros((len(pdb_atoms), len(ele2num)), dtype=np.float32)
                for i, t in enumerate(pdb_types):
                    types_array[i, t] = 1.0

                np.savez(
                    processed_file,
                    atoms = coords,
                    types = types_array
                )

    elif args.dataset == 'pdbbind':
        with open(os.path.join(root, "metadata/affinities.json"), "r") as f:
            affinity_dict = json.load(f)
        with open(os.path.join(root, "metadata/lig_smiles.json"), "r") as f:
            lig_smiles = json.load(f)
        protein_names = list(affinity_dict.keys())


        for protein_name in tqdm(protein_names):
            pdb_file = os.path.join(pdb_files, protein_name)
            if args.pdb_fix:
                pdb_root = os.path.join(pdb_file, protein_name + '_fixed.pdb')
                if not os.path.exists(pdb_root):
                    pdb_root = os.path.join(pdb_file, protein_name + '.pdb')
            else:
                pdb_root = os.path.join(pdb_file, protein_name + '.pdb')

            output_file = os.path.join(output_files, protein_name)
            os.makedirs(output_file, exist_ok=True)
            processed_file = os.path.join(output_file, protein_name + '.npz')

            if args.surface:
                parser = PDBParser()
                structure = parser.get_structure("structure", pdb_root)
                atoms = structure.get_atoms()

                pdb_atoms = []
                pdb_types = []

                for atom in atoms:
                    pdb_atoms.append(atom.get_coord())
                    if atom.element in ele2num:
                        pdb_types.append(ele2num[atom.element])
                    else:
                        pdb_types.append(ele2num['OTH'])

                coords = np.stack(pdb_atoms)
                types_array = np.zeros((len(pdb_atoms), len(ele2num)), dtype=np.float32)
                for i, t in enumerate(pdb_types):
                    types_array[i, t] = 1.0

                np.savez(
                    processed_file,
                    atoms=coords,
                    types=types_array
                )

    elif args.dataset == 'ec':
        pdb_files = os.path.join(root, 'EnzymeCommission', split)
        fasta_file = os.path.join(root, split + '.fasta')

        with open(fasta_file, 'r') as f:
            protein_names = []
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                    protein_names.append(protein_name.replace('.', '_'))
                else:
                    pass

        raw_names = os.listdir(pdb_files)

        for protein_name in tqdm(protein_names):
            try:
                pdb_file = pdb_files
                if args.pdb_fix:
                    pdb_root = os.path.join(pdb_file, protein_name + '_fixed.pdb')
                    if not os.path.exists(pdb_root):
                        pdb_root = os.path.join(pdb_file, protein_name + '.pdb')
                else:
                    count = 0
                    selected_protein = ''
                    for raw_name in raw_names:
                        if protein_name in raw_name:
                            selected_protein += raw_name
                            count += 1
                        assert count < 2
                    pdb_root = os.path.join(pdb_file, selected_protein)

                output_file = os.path.join(output_files, protein_name)
                os.makedirs(output_file, exist_ok=True)
                processed_file = os.path.join(output_file, protein_name + '.npz')

                if args.surface:
                    parser = PDBParser()
                    structure = parser.get_structure("structure", pdb_root)
                    atoms = structure.get_atoms()

                    pdb_atoms = []
                    pdb_types = []

                    for atom in atoms:
                        pdb_atoms.append(atom.get_coord())
                        if atom.element in ele2num:
                            pdb_types.append(ele2num[atom.element])
                        else:
                            pdb_types.append(ele2num['OTH'])

                    coords = np.stack(pdb_atoms)
                    types_array = np.zeros((len(pdb_atoms), len(ele2num)), dtype=np.float32)
                    for i, t in enumerate(pdb_types):
                        types_array[i, t] = 1.0

                    np.savez(
                        processed_file,
                        atoms=coords,
                        types=types_array
                    )

            except Exception as e:
                print('error', protein_name)
                print(e)
                continue

    elif args.dataset == 'go':
        pdb_files = os.path.join(root, 'GeneOntology', split)
        fasta_file = os.path.join(root, split + '.fasta')

        protein_seqs = []
        with open(fasta_file, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    amino_chain = line.rstrip()
                    protein_seqs.append((protein_name, amino_chain))

        raw_names = os.listdir(pdb_files)

        for protein_name, amino_chain in tqdm(protein_seqs):
            # try:
            pdb_file = pdb_files
            if args.pdb_fix:
                pdb_root = os.path.join(pdb_file, protein_name + '_fixed.pdb')
                if not os.path.exists(pdb_root):
                    pdb_root = os.path.join(pdb_file, protein_name + '.pdb')
            else:
                count = 0
                selected_protein = ''
                for raw_name in raw_names:
                    if protein_name in raw_name:
                        selected_protein += raw_name
                        count += 1
                    assert count < 2
                pdb_root = os.path.join(pdb_file, selected_protein)

            output_file = os.path.join(output_files, protein_name)
            os.makedirs(output_file, exist_ok=True)
            processed_file = os.path.join(output_file, protein_name + '.npz')

            if args.surface:
                parser = PDBParser()
                structure = parser.get_structure("structure", pdb_root)
                # parser = MMCIFParser()
                # structure = parser.get_structure("structure", pdb_root)
                atoms = structure.get_atoms()

                pdb_atoms = []
                pdb_types = []

                for atom in atoms:
                    pdb_atoms.append(atom.get_coord())
                    if atom.element in ele2num:
                        pdb_types.append(ele2num[atom.element])
                    else:
                        pdb_types.append(ele2num['OTH'])

                coords = np.stack(pdb_atoms)
                types_array = np.zeros((len(pdb_atoms), len(ele2num)), dtype=np.float32)
                for i, t in enumerate(pdb_types):
                    types_array[i, t] = 1.0

                np.savez(
                    processed_file,
                    atoms=coords,
                    types=types_array
                )

            # except Exception as e:
            #     print('error', protein_name)
            #     print(e)
            #     continue



    print('processed ' + split + ' set')






if __name__ == '__main__':
    args = parse_args()
    # for split in ['training', 'validation', 'testing']:
    for split in ['train', 'valid', 'test']:
    # for split in ['training', 'validation', 'test_fold', 'test_family', 'test_superfamily']:
        main(args, split)

    print('Finish')
















