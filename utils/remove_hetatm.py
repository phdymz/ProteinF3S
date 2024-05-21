# remove hetatm atoms in pdb, make pdb loaded in torchdrug
import os
from tqdm import tqdm




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
        pdb_file = os.path.join(pdb_files, protein_name)
        pdb_root = os.path.join(pdb_file, protein_name + '.pdb')

        new_pdb = ''
        exist_X = False

        with open(pdb_root, 'r') as f:
            lines = f.readlines()
            f.close()
            for line in lines:
                if line[77] == 'X':
                    exist_X = True
                    break
                else:
                    new_pdb += line

            if exist_X:
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




















