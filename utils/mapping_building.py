import numpy as np
import torch
import os
from tqdm import tqdm
from torchdrug import data
import utils.load_pdb
import json
from datasets.builder import batch_neighbors_kpconv, downsample_struct
import open3d as o3d


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Data_Pre_Process')
    parser.add_argument('--dataset', default='go', type=str,
                        choices=['func', 'ec', 'pdbbind', 'fold', 'go'])
    parser.add_argument('--data_dir', default='/media/ymz/807a6847-34c1-4630-bb53-2a277f79be0c/ProteinF3S/go', type=str,
                        metavar='N', help='data root directory')
    parser.add_argument('--pdb_fix', default=False, type=bool)
    parser.add_argument('--surface', default=False, type=bool)
    parser.add_argument('--sequence', default=False, type=bool)

    parser.add_argument('--K', default=16, type=int)

    args = parser.parse_args()
    return args


def main(args, split):
    if args.dataset == 'func':
        root = args.data_dir
        output_files = os.path.join(root, 'mapping')

        fasta_file = os.path.join(root, 'chain_'+split+'.fasta')

        with open(fasta_file, 'r') as f:
            protein_names = []
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                    protein_names.append(protein_name.replace('.', '_'))
                else:
                    pass
        # protein_names = ['2qe7_G', '2qe7_H']
        if split == 'training':
            protein_names.remove('2qe7_G')
            protein_names.remove('2qe7_H')

        for protein_name in tqdm(protein_names):
            # try:
            struct = os.path.join(root, 'structure', protein_name, protein_name + '.npz')
            surf = os.path.join(root, 'surface', protein_name, protein_name + '.npz')
            processed_file = os.path.join(root, 'mapping', protein_name + '.npz')

            struct = np.load(struct)
            surf = np.load(surf)

            batched_pos = torch.from_numpy(struct['coords'])
            batched_points = torch.from_numpy(surf['xyz'])
            seq = np.expand_dims(a=np.arange(batched_pos.shape[0]), axis=1).astype(dtype=np.float32)
            batched_seq = torch.from_numpy(seq)
            batched_index = 0 * torch.ones(len(batched_seq), dtype=torch.int64)

            batched_lengths_struct = torch.tensor([batched_pos.shape[0]]).int()
            batched_lengths = torch.tensor([batched_points.shape[0]]).int()

            radius = 6.0

            surf2struct = batch_neighbors_kpconv(batched_pos, batched_points, batched_lengths_struct, batched_lengths,
                                   radius, args.K).long()
            struct2surf = batch_neighbors_kpconv(batched_points, batched_pos, batched_lengths, batched_lengths_struct,
                                                                        radius, 1).long()

            batched_pos, batched_seq, batched_index = downsample_struct(batched_pos, batched_seq, batched_index)

            if (struct['coords'].max(0) > surf['xyz'].max(0)).sum() > 0 or (surf['xyz'].min(0) > struct['coords'].min(0)).sum() > 0:
                print(protein_name)
                pcd0 = o3d.geometry.PointCloud()
                pcd1 = o3d.geometry.PointCloud()
                pcd2 = o3d.geometry.PointCloud()
                pcd0.points = o3d.utility.Vector3dVector(struct['coords'])
                pcd1.points = o3d.utility.Vector3dVector(surf['xyz'])
                pcd2.points = o3d.utility.Vector3dVector(np.load(os.path.join(root, 'atom', protein_name, protein_name + '.npz'))['atoms'])
                pcd0.paint_uniform_color([1, 0, 0])
                pcd1.paint_uniform_color([0, 0, 1])
                pcd2.paint_uniform_color([0, 1, 0])
                o3d.visualization.draw_geometries([pcd0, pcd1, pcd2])



            np.savez(
                processed_file,
                surf2struct = surf2struct,
                struct2surf = struct2surf,
                batched_pos = batched_pos,
                batched_seq = batched_seq
            )
            # except Exception as e:
            #     print(protein_name)

    if args.dataset == 'ec':
        root = args.data_dir
        output_files = os.path.join(root, 'mapping')

        fasta_file = os.path.join(root, split+'.fasta')

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
                struct = os.path.join(root, 'structure', protein_name, protein_name + '.npz')
                surf = os.path.join(root, 'surface', protein_name, protein_name + '.npz')
                processed_file = os.path.join(root, 'mapping', protein_name + '.npz')

                struct = np.load(struct)
                surf = np.load(surf)

                batched_pos = torch.from_numpy(struct['coords'])
                batched_points = torch.from_numpy(surf['xyz'])
                seq = np.expand_dims(a=np.arange(batched_pos.shape[0]), axis=1).astype(dtype=np.float32)
                batched_seq = torch.from_numpy(seq)
                batched_index = 0 * torch.ones(len(batched_seq), dtype=torch.int64)

                batched_lengths_struct = torch.tensor([batched_pos.shape[0]]).int()
                batched_lengths = torch.tensor([batched_points.shape[0]]).int()

                radius = 6.0

                surf2struct = batch_neighbors_kpconv(batched_pos, batched_points, batched_lengths_struct, batched_lengths,
                                       radius, args.K).long()
                struct2surf = batch_neighbors_kpconv(batched_points, batched_pos, batched_lengths, batched_lengths_struct,
                                                                            radius, 1).long()

                batched_pos, batched_seq, batched_index = downsample_struct(batched_pos, batched_seq, batched_index)

                if (struct['coords'].max(0) > surf['xyz'].max(0)).sum() > 0 or (surf['xyz'].min(0) > struct['coords'].min(0)).sum() > 0:
                    print(protein_name)
                    pcd0 = o3d.geometry.PointCloud()
                    pcd1 = o3d.geometry.PointCloud()
                    pcd2 = o3d.geometry.PointCloud()
                    pcd0.points = o3d.utility.Vector3dVector(struct['coords'])
                    pcd1.points = o3d.utility.Vector3dVector(surf['xyz'])
                    pcd2.points = o3d.utility.Vector3dVector(np.load(os.path.join(root, 'atom', protein_name, protein_name + '.npz'))['atoms'])
                    pcd0.paint_uniform_color([1, 0, 0])
                    pcd1.paint_uniform_color([0, 0, 1])
                    pcd2.paint_uniform_color([0, 1, 0])
                    o3d.visualization.draw_geometries([pcd0, pcd1, pcd2])



                np.savez(
                    processed_file,
                    surf2struct = surf2struct,
                    struct2surf = struct2surf,
                    batched_pos = batched_pos,
                    batched_seq = batched_seq
                )
            except Exception as e:
                print('error', protein_name, e)
                continue

    if args.dataset == 'go':
        root = args.data_dir

        fasta_file = os.path.join(root, split+'.fasta')

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
                struct = os.path.join(root, 'structure', protein_name, protein_name + '.npz')
                surf = os.path.join(root, 'surface', protein_name, protein_name + '.npz')
                processed_file = os.path.join(root, 'mapping', protein_name + '.npz')

                struct = np.load(struct)
                surf = np.load(surf)

                batched_pos = torch.from_numpy(struct['coords'])
                batched_points = torch.from_numpy(surf['xyz'])
                seq = np.expand_dims(a=np.arange(batched_pos.shape[0]), axis=1).astype(dtype=np.float32)
                batched_seq = torch.from_numpy(seq)
                batched_index = 0 * torch.ones(len(batched_seq), dtype=torch.int64)

                batched_lengths_struct = torch.tensor([batched_pos.shape[0]]).int()
                batched_lengths = torch.tensor([batched_points.shape[0]]).int()

                radius = 6.0

                surf2struct = batch_neighbors_kpconv(batched_pos, batched_points, batched_lengths_struct, batched_lengths,
                                       radius, args.K).long()
                struct2surf = batch_neighbors_kpconv(batched_points, batched_pos, batched_lengths, batched_lengths_struct,
                                                                            radius, 1).long()

                batched_pos, batched_seq, batched_index = downsample_struct(batched_pos, batched_seq, batched_index)

                if (struct['coords'].max(0) > surf['xyz'].max(0)).sum() > 0 or (surf['xyz'].min(0) > struct['coords'].min(0)).sum() > 0:
                    print(protein_name)
                    pcd0 = o3d.geometry.PointCloud()
                    pcd1 = o3d.geometry.PointCloud()
                    pcd2 = o3d.geometry.PointCloud()
                    pcd0.points = o3d.utility.Vector3dVector(struct['coords'])
                    pcd1.points = o3d.utility.Vector3dVector(surf['xyz'])
                    pcd2.points = o3d.utility.Vector3dVector(np.load(os.path.join(root, 'atom', protein_name, protein_name + '.npz'))['atoms'])
                    pcd0.paint_uniform_color([1, 0, 0])
                    pcd1.paint_uniform_color([0, 0, 1])
                    pcd2.paint_uniform_color([0, 1, 0])
                    o3d.visualization.draw_geometries([pcd0, pcd1, pcd2])

                    print('mis_align', protein_name)



                np.savez(
                    processed_file,
                    surf2struct = surf2struct,
                    struct2surf = struct2surf,
                    batched_pos = batched_pos,
                    batched_seq = batched_seq
                )
            except Exception as e:
                print('error', protein_name, e)
                continue

    print('processed ' + split + ' set')


if __name__ == '__main__':
    args = parse_args()
    # for split in ['training', 'validation', 'testing']:
    for split in ['train', 'valid', 'test']:
        main(args, split)

    print('Finish')


















