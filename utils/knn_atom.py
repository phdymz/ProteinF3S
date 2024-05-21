import numpy as np
import torch
import os
from tqdm import tqdm
from pytorch3d.ops.knn import knn_points



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
    output_files = os.path.join(root, 'surface')

    # fasta_file = os.path.join(root, 'chain_'+split+'.fasta')
    fasta_file = os.path.join(root, split + '.fasta')

    with open(fasta_file, 'r') as f:
        protein_names = []
        for line in f:
            if line.startswith('>'):
                protein_name = line.rstrip()[1:]
                protein_names.append(protein_name.replace('.', '_'))
            else:
                pass

    for protein_name in tqdm(protein_names):
        try:
            output_file = os.path.join(output_files, protein_name)
            os.makedirs(output_file, exist_ok=True)
            processed_file = os.path.join(output_file, protein_name + '.npz')

            if args.surface:
                data = np.load(processed_file)
                xyz = torch.from_numpy(data['xyz'])
                normal = data['normal']
                curvature = data['curvature']
                atom = torch.from_numpy(data['atom'])
                atom_type = torch.from_numpy(data['atom_type'])

                dists, idx, _ = knn_points(xyz.unsqueeze(0), atom.unsqueeze(0), K=16)
                dists = dists.squeeze(0)
                idx = idx.squeeze(0)
                atom_type_sel = atom_type[idx]
                chem = torch.cat([atom_type_sel, dists.unsqueeze(-1)], dim=-1)

                np.savez(
                    processed_file,
                    xyz = xyz.numpy(),
                    normal = normal,
                    curvature = curvature,
                    atom = atom.numpy(),
                    atom_type = atom_type.numpy(),
                    chem = chem.numpy()
                )

        except Exception as e:
            print('error', e)
            continue


    print('processed ' + split + ' set')


if __name__ == '__main__':

    args = parse_args()
    # for split in ['training', 'validation', 'testing']:
    for split in ['train', 'valid', 'test']:
        main(args, split)

    print('Finish')


















