import numpy as np
import torch
import os
from tqdm import tqdm
from utils.show_surface import construct_surface
from utils.Arguments import parser
import json



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Data_Pre_Process')
    parser.add_argument('--dataset', default='go', type=str,
                        choices=['func', 'ec', 'pdbbind', 'fold'])
    parser.add_argument('--data_dir', default='/media/ymz/807a6847-34c1-4630-bb53-2a277f79be0c/ProteinF3S/go', type=str,
                        metavar='N', help='data root directory')
    parser.add_argument('--pdb_fix', default=False, type=bool)
    parser.add_argument('--surface', default=True, type=bool)
    parser.add_argument('--sequence', default=False, type=bool)

    args = parser.parse_args()
    return args


def main(args, split, cfg):

    root = args.data_dir
    output_files = os.path.join(root, 'surface')
    if args.pdb_fix:
        output_files = os.path.join(root, 'surface_fixed')
    pdb_files = os.path.join(root, 'pdb_files')

    if args.dataset == 'func':
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

            if args.surface:
                xyz, normal, curvature, atom, type = construct_surface(
                    pdb_root.replace('pdb_files', 'atom').replace('.pdb', '.npz'), cfg)

                np.savez(
                    processed_file,
                    xyz = xyz,
                    normal = normal,
                    curvature = curvature,
                    atom = atom,
                    atom_type = type
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
                try:
                    xyz, normal, curvature, atom, type = construct_surface(
                        pdb_root.replace('pdb_files', 'atom').replace('.pdb', '.npz'), cfg)

                    np.savez(
                        processed_file,
                        xyz=xyz,
                        normal=normal,
                        curvature=curvature,
                        atom=atom,
                        atom_type=type
                    )
                except Exception as e:
                    print(e)
                    continue

    elif args.dataset == 'ec':
        fasta_file = os.path.join(root, split + '.fasta')
        pdb_files = os.path.join(root, 'EnzymeCommission', split)

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
                pdb_file = os.path.join(pdb_files, protein_name)
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
                    xyz, normal, curvature, atom, type = construct_surface(
                        os.path.join(root, 'atom', protein_name, protein_name + '.npz'), cfg)

                    np.savez(
                        processed_file,
                        xyz=xyz,
                        normal=normal,
                        curvature=curvature,
                        atom=atom,
                        atom_type=type
                    )

            except Exception as e:
                print('error', protein_name)
                print(e)
                continue

    elif args.dataset == 'go':
        fasta_file = os.path.join(root, split + '.fasta')
        pdb_files = os.path.join(root, 'GeneOntology', split)

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
                pdb_file = os.path.join(pdb_files, protein_name)
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
                    xyz, normal, curvature, atom, type = construct_surface(
                        os.path.join(root, 'atom', protein_name, protein_name + '.npz'), cfg)

                    np.savez(
                        processed_file,
                        xyz=xyz,
                        normal=normal,
                        curvature=curvature,
                        atom=atom,
                        atom_type=type
                    )

            except Exception as e:
                print('error', protein_name)
                print(e)
                continue


    print('processed ' + split + ' set')


if __name__ == '__main__':
    cfg = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)

    args = parse_args()
    # for split in ['training', 'validation', 'testing']:
    for split in ['train', 'valid', 'test']:
        main(args, split, cfg)

    print('Finish')


















