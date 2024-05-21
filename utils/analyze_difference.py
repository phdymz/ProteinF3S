import numpy as np
import torch
import os
from tqdm import tqdm
from torchdrug import data
import utils.load_pdb


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
    parser.add_argument('--dataset', default='func', type=str,
                        choices=['func', 'ec', 'pdbbind', 'fold'])
    parser.add_argument('--data_dir', default='/media/ymz/807a6847-34c1-4630-bb53-2a277f79be0c/ProteinF3S/func', type=str,
                        metavar='N', help='data root directory')
    parser.add_argument('--pdb_fix', default=False, type=bool)
    parser.add_argument('--surface', default=False, type=bool)
    parser.add_argument('--sequence', default=False, type=bool)

    args = parser.parse_args()
    return args


def main(args, split):

    not_match = []
    match= []

    root = args.data_dir
    output_files = os.path.join(root, 'processed_protein')
    pdb_files = os.path.join(root, 'pdb_files')

    fasta_file = os.path.join(root, 'chain_'+split+'.fasta')

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

        pdb = np.load(processed_file)
        coords = pdb['coords']
        amino_ids = pdb['amino_ids']


        coords_fhh = np.load(os.path.join('/media/ymz/807a6847-34c1-4630-bb53-2a277f79be0c/CDConv/func/coordinates/training', protein_name.replace('_','.') + '.npy'))
        if coords_fhh.shape != coords.shape:
            not_match.append((coords_fhh.shape, coords.shape))
        else:
            match.append(((coords_fhh - coords)**2).mean())


        if args.surface:
            pass

        if args.sequence:
            pass


    print('processed ' + split + ' set')


if __name__ == '__main__':
    args = parse_args()
    for split in ['training', 'validation', 'testing']:
        main(args, split)

    print('Finish')


