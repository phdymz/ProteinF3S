import numpy as np
import torch
import os
from tqdm import tqdm
import torchdrug
import utils.load_pdb_seq
import json

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
    parser.add_argument('--surface', default=False, type=bool)
    parser.add_argument('--sequence', default=True, type=bool)

    args = parser.parse_args()
    return args


def main(args, split):
    if args.dataset == 'func':
        root = args.data_dir
        output_files = os.path.join(root, 'sequence')
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
            processed_file = os.path.join(output_file, protein_name + '.npy')


            pdb = utils.load_pdb_seq.Protein_New_Seq.from_pdb(pdb_root, atom_feature=None, bond_feature=None,
                                                          residue_feature=None)

            idx = pdb.atom_name == pdb.atom_name2id['CA']
            idx = pdb.atom2residue[idx]
            amino_ids = pdb.residue_type[idx].numpy()

            amino_seq = ''
            for amino_id in amino_ids:
                amino_seq += pdb.id2residue_symbol[amino_id]

            if args.sequence:
                np.save(
                    processed_file,
                    np.array(amino_seq)
                )

    elif args.dataset == 'pdbbind':
        root = args.data_dir
        if not args.pdb_fix:
            output_files = os.path.join(root, 'sequence_fixed')
        else:
            output_files = os.path.join(root, 'sequence')
        pdb_files = os.path.join(root, 'pdb_files')

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
            processed_file = os.path.join(output_file, protein_name + '.npy')

            pdb = utils.load_pdb_seq.Protein_New_Seq.from_pdb(pdb_root, atom_feature=None, bond_feature=None,
                                                              residue_feature=None)

            if pdb is None:
                print(protein_name + ' could not processed')
                continue

            idx = pdb.atom_name == pdb.atom_name2id['CA']
            idx = pdb.atom2residue[idx]
            amino_ids = pdb.residue_type[idx].numpy()

            amino_seq = ''
            for amino_id in amino_ids:
                amino_seq += pdb.id2residue_symbol[amino_id]

            if args.sequence:
                np.save(
                    processed_file,
                    np.array(amino_seq)
                )

    elif args.dataset == 'ec':
        root = args.data_dir
        output_files = os.path.join(root, 'sequence')
        pdb_files = os.path.join(root, 'EnzymeCommission', split)

        fasta_file = os.path.join(root, split + '.fasta')

        protein_seqs = []
        with open(fasta_file, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    amino_chain = line.rstrip()
                    protein_seqs.append((protein_name, amino_chain))

        for protein_name, amino_ids_cdc in tqdm(protein_seqs):

            output_file = os.path.join(output_files, protein_name)
            os.makedirs(output_file, exist_ok=True)
            processed_file = os.path.join(output_file, protein_name + '.npy')

            if args.sequence:
                np.save(
                    processed_file,
                    np.array(amino_ids_cdc)
                )

    elif args.dataset == 'fold':
        root = args.data_dir
        output_files = os.path.join(root, 'sequence')

        fasta_file = os.path.join(root, split + '.fasta')

        protein_seqs = []
        with open(fasta_file, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    amino_chain = line.rstrip()
                    protein_seqs.append((protein_name, amino_chain))

        for protein_name, amino_ids_cdc in tqdm(protein_seqs):

            output_file = os.path.join(output_files, protein_name)
            os.makedirs(output_file, exist_ok=True)
            processed_file = os.path.join(output_file, protein_name + '.npy')

            if args.sequence:
                np.save(
                    processed_file,
                    np.array(amino_ids_cdc)
                )

    elif args.dataset == 'go':
        root = args.data_dir
        output_files = os.path.join(root, 'sequence')

        fasta_file = os.path.join(root, split + '.fasta')

        protein_seqs = []
        with open(fasta_file, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    amino_chain = line.rstrip()
                    protein_seqs.append((protein_name, amino_chain))

        for protein_name, amino_ids_cdc in tqdm(protein_seqs):

            output_file = os.path.join(output_files, protein_name)
            os.makedirs(output_file, exist_ok=True)
            processed_file = os.path.join(output_file, protein_name + '.npy')

            if args.sequence:
                np.save(
                    processed_file,
                    np.array(amino_ids_cdc)
                )



    print('processed ' + split + ' set')


if __name__ == '__main__':
    args = parse_args()
    # for split in ['training', 'validation', 'testing']:
    for split in ['train', 'valid', 'test']:
    # for split in ['training', 'validation', 'test_fold', 'test_family', 'test_superfamily']:
        main(args, split)

    print('Finish')


