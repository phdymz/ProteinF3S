# pdb files provided by holoprot make mistake in protein chain

import os
from tqdm import tqdm

alphabet2id = {" ": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10,
                   "K": 11, "L": 12, "M": 13, "N": 14, "O": 15, "P": 16, "Q": 17, "R": 18, "S": 19, "T": 20,
                   "U": 21, "V": 22, "W": 23, "X": 24, "Y": 25, "Z": 26}


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Data_Pre_Process')
    parser.add_argument('--dataset', default='func', type=str,
                        choices=['func', 'ec', 'pdbbind', 'fold'])
    parser.add_argument('--data_dir', default='/media/ymz/807a6847-34c1-4630-bb53-2a277f79be0c/ProteinF3S/func', type=str,
                        metavar='N', help='data root directory')

    args = parser.parse_args()
    return args


def main(args, split):

    count = 0

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
        chain = protein_name.split('_')[1]
        if chain in alphabet2id:
            continue
        pdb_file = os.path.join(pdb_files, protein_name)
        pdb_root = os.path.join(pdb_file, protein_name + '.pdb')

        new_pdb = ''
        chain = 'A'

        with open(pdb_root, 'r') as f:
            lines = f.readlines()
            f.close()
            for line in lines:
                line_new = line[:21] + chain + line[22:]
                new_pdb += line_new


            print(protein_name)
            count+=1
            with open(pdb_root, 'w') as f:
                f.write(new_pdb)
                f.close()

            del new_pdb




    print(count)
    print('processed ' + split + ' set')


if __name__ == '__main__':
    args = parse_args()
    for split in ['training', 'validation', 'testing']:
        main(args, split)

    print('Finish')




















