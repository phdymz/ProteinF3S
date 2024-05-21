import numpy as np
import torch
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import normalize
from scipy.spatial.transform import Rotation
import json


def orientation(pos):
    u = normalize(X=pos[1:,:] - pos[:-1,:], norm='l2', axis=1)
    u1 = u[1:,:]
    u2 = u[:-1, :]
    b = normalize(X=u2 - u1, norm='l2', axis=1)
    n = normalize(X=np.cross(u2, u1), norm='l2', axis=1)
    o = normalize(X=np.cross(b, n), norm='l2', axis=1)
    ori = np.stack([b, n, o], axis=1)
    return np.concatenate([np.expand_dims(ori[0], 0), ori, np.expand_dims(ori[-1], 0)], axis=0)



class PDBBindDataset(Dataset):
    def __init__(self, cfg, root='/media/ymz/807a6847-34c1-4630-bb53-2a277f79be0c/ProteinF3S/pdbbind', random_seed=0,
                 split='training', rotation = True):

        self.random_state = np.random.RandomState(random_seed)
        self.split = split
        self.rotation = rotation
        self.cfg = cfg

        self.surface = self.cfg.surface
        self.structure = self.cfg.structure
        self.sequence = self.cfg.sequence

        # Get the paths.
        npy_dir = os.path.join(root, 'sequence')
        self.npy_dir = npy_dir

        # Load the fasta file.
        with open(os.path.join(root, "metadata/affinities.json"), "r") as f:
            affinity_dict = json.load(f)

        self.affinity_dict = affinity_dict

        with open(f"{root}/metadata/{self.cfg.spliting}.json", "r") as f:
            data_splits = json.load(f)
        if split == 'training':
            self.mode = 'train'
        elif split == 'validation':
            self.mode = 'valid'
        elif split == 'testing':
            self.mode = 'test'
        self.protein_names = data_splits[self.mode]
        self.check_pdbs()
        self.ligand = True

        print()



    def __len__(self):
        return len(self.protein_names)

    def __getitem__(self, idx):
        data = self.protein_names[idx]
        data_dir = os.path.join(self.npy_dir, data.replace('.', '_'), data.replace('.', '_') + '.npz')
        label = self.affinity_dict[data]

        output = {}
        rot = self.gen_rot()

        if self.sequence:
            data_dir_sequence = data_dir
            amino_sequence = np.load(data_dir_sequence.replace('.npz', '.npy'))
            amino_sequence = str(amino_sequence)
            output['sequence'] = ('', amino_sequence)

        if self.structure:
            data_dir_structure = data_dir.replace('sequence', 'structure')
            data = np.load(data_dir_structure)
            pos = data['coords']
            amino_ids = data['amino_ids']
            center = np.sum(a=pos, axis=0, keepdims=True) / pos.shape[0]
            pos = pos - center
            if self.rotation:
                pos = np.matmul(pos, rot.numpy())
            ori = orientation(pos)
            amino = amino_ids.astype(int)

            if self.split == "training":
                pos = pos + self.random_state.normal(0.0, 0.05, pos.shape)

            pos = pos.astype(dtype=np.float32)
            ori = ori.astype(dtype=np.float32)
            seq = np.expand_dims(a=np.arange(pos.shape[0]), axis=1).astype(dtype=np.float32)

            x = torch.from_numpy(amino)
            ori = torch.from_numpy(ori)
            seq=torch.from_numpy(seq)
            pos=torch.from_numpy(pos)

            output['x'] = x
            output['ori'] = ori
            output['seq'] = seq
            output['pos'] = pos

        if self.surface:
            data_dir_surface = data_dir.replace('sequence', 'surface')
            data = np.load(data_dir_surface)

            if not self.structure:
                center = np.load(data_dir_surface.replace('surface', 'structure'))['coords'].mean(0)

            xyz = torch.from_numpy(data['xyz'])
            # normal = torch.from_numpy(data['normal'])
            curvature = torch.from_numpy(data['curvature'])
            chem = torch.from_numpy(data['chem'])

            xyz = xyz - torch.from_numpy(center)
            if self.rotation:
                xyz = torch.matmul(xyz, rot)

            if self.split == "training":
                xyz = xyz + self.random_state.normal(0.0, 0.05, xyz.shape).astype(np.float32)

            output['point'] = xyz
            output['geo'] = curvature
            output['chem'] = chem
        output['label'] = torch.tensor(label).reshape(-1)

        if self.cfg.fusion_type == 'msf':
            data = np.load(os.path.join(self.cfg.data_dir, 'mapping', self.protein_names[idx].replace('.', '_') + '.npz'))
            output['surf2struct'] = torch.from_numpy(data['surf2struct'])
            output['struct2surf'] = torch.from_numpy(data['struct2surf'])
            if self.rotation:
                output['batched_pos'] = torch.matmul(torch.from_numpy(data['batched_pos'])  - center, rot)
            else:
                output['batched_pos'] = torch.from_numpy(data['batched_pos'])  - center
            output['batched_seq'] = torch.from_numpy(data['batched_seq'])

        if self.ligand:
            data_dir_ligand = data_dir.replace('sequence', 'ligand').replace('.npz', '.pth')
            data = torch.load(data_dir_ligand)
            output['ligand'] = data

        return output

    def gen_rot(self):
        R = torch.FloatTensor(Rotation.random().as_matrix())
        return R

    def check_pdbs(self):
        null_list = []
        for protein_name in self.protein_names:
            path_pdb = os.path.join(self.npy_dir, protein_name, protein_name + '.npy')
            if not os.path.exists(path_pdb):
                null_list.append(protein_name)
        for item in null_list:
            self.protein_names.remove(item)




if __name__ == '__main__':
    PDBBindDataset()
    print()































