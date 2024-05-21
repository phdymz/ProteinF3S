






















import numpy as np
import torch
import os
from tqdm import tqdm
from torchdrug import data
import utils.load_pdb
import json

def get_coords(residues, mode = 'ca'):
    if mode == 'ca':
        coords = np.array([res['CA'].get_coord() for res in residues])
    elif mode == 'com':
        coords = None
    else:
        raise ValueError(f'Coords cannot be generated for mode {mode}')
    return coords




id2residue_symbol = {0: "G", 1: "A", 2: "S", 3: "P", 4: "V", 5: "T", 6: "C", 7: "I", 8: "L", 9: "N",
                         10: "D", 11: "Q", 12: "K", 13: "E", 14: "M", 15: "H", 16: "F", 17: "R", 18: "Y", 19: "W", 20: "X"}
residue_symbol2id = {v: k for k, v in id2residue_symbol.items()}

aa = "ACDEFGHIKLMNPQRSTVWYX"
aa_to_id = {}
for i in range(0, 21):
    aa_to_id[aa[i]] = i


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


    if args.dataset == 'go':
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

                pdb = utils.load_pdb.Protein_New.from_pdb(pdb_root, atom_feature=None, bond_feature=None,
                                                          residue_feature=None)

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


    print('processed ' + split + ' set')


if __name__ == '__main__':
    args = parse_args()
    # for split in ['training', 'validation', 'test_fold', 'test_family', 'test_superfamily']:
    for split in ['train', 'valid', 'test']:
        main(args, split)

    print('Finish')


