import numpy as np
import torch
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import normalize
from tqdm import tqdm
from pytorch3d.ops.knn import knn_points
from scipy.spatial.transform import Rotation
from datasets.builder import get_dataloader



def orientation(pos):
    u = normalize(X=pos[1:,:] - pos[:-1,:], norm='l2', axis=1)
    u1 = u[1:,:]
    u2 = u[:-1, :]
    b = normalize(X=u2 - u1, norm='l2', axis=1)
    n = normalize(X=np.cross(u2, u1), norm='l2', axis=1)
    o = normalize(X=np.cross(b, n), norm='l2', axis=1)
    ori = np.stack([b, n, o], axis=1)
    return np.concatenate([np.expand_dims(ori[0], 0), ori, np.expand_dims(ori[-1], 0)], axis=0)


aa = "ACDEFGHIKLMNPQRSTVWYX"
aa_to_id = {}
for i in range(0, 21):
    aa_to_id[aa[i]] = i


class GODataset(Dataset):
    def __init__(self, cfg, root='/media/ymz/807a6847-34c1-4630-bb53-2a277f79be0c/ProteinF3S/go', level='mf',percent=30, random_seed=0,
                 split='train', rotation = True):

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
        fasta_file = os.path.join(root, split+'.fasta')

        # Mask test set.
        test_set = set()
        if split == "test":
            with open(os.path.join(root, "nrPDB-GO_2019.06.18_test.csv"), 'r') as f:
                head = True
                for line in f:
                    if head:
                        head = False
                        continue
                    arr = line.rstrip().split(',')
                    if percent == 30 and arr[1] == '1':
                        test_set.add(arr[0])
                    elif percent == 40 and arr[2] == '1':
                        test_set.add(arr[0])
                    elif percent == 50 and arr[3] == '1':
                        test_set.add(arr[0])
                    elif percent == 70 and arr[4] == '1':
                        test_set.add(arr[0])
                    elif percent == 95 and arr[5] == '1':
                        test_set.add(arr[0])
                    else:
                        pass

        # Load the fasta file.
        protein_names = []
        with open(fasta_file, 'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    if split == "test" and (protein_name not in test_set):
                        continue
                    protein_names.append(protein_name)

        level_idx = 0
        go_cnt = 0
        go_num = {}
        go_annotations = {}
        self.labels = {}
        with open(os.path.join(root, 'nrPDB-GO_2019.06.18_annot.tsv'), 'r') as f:
            for idx, line in enumerate(f):
                if idx == 1 and level == "mf":
                    level_idx = 1
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx == 5 and level == "bp":
                    level_idx = 2
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx == 9 and level == "cc":
                    level_idx = 3
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx > 12:
                    arr = line.rstrip().split('\t')
                    protein_labels = []
                    if len(arr) > level_idx:
                        protein_go_list = arr[level_idx]
                        protein_go_list = protein_go_list.split(',')
                        for go in protein_go_list:
                            if len(go) > 0:
                                protein_labels.append(go_annotations[go])
                                go_num[go] += 1
                    self.labels[arr[0]] = np.array(protein_labels)

        self.num_classes = len(go_annotations)
        self.protein_names = protein_names

        self.weights = np.zeros((go_cnt,), dtype=np.float32)
        for go, idx in go_annotations.items():
            self.weights[idx] = len(self.labels) / go_num[go]


        if self.surface:
            surface_data = os.listdir(os.path.join(root, 'surface'))
            for item in surface_data:
                surf_path = os.path.join(root, 'surface', item, item + '.npz')
                if not os.path.exists(surf_path):
                    if item in self.protein_names:
                        self.protein_names.remove(item)



    def __len__(self):
        return len(self.protein_names)

    def __getitem__(self, idx):
        data = self.protein_names[idx]
        data_dir = os.path.join(self.npy_dir, data.replace('.', '_'), data.replace('.', '_') + '.npz')

        label = np.zeros((self.num_classes,)).astype(np.float32)
        if len(self.labels[data]) > 0:
            label[self.labels[data]] = 1.0

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
        output['label'] = torch.tensor(label).reshape(1, -1)

        if self.cfg.fusion_type == 'msf':
            data = np.load(os.path.join(self.cfg.data_dir, 'mapping', self.protein_names[idx].replace('.', '_') + '.npz'))
            output['surf2struct'] = torch.from_numpy(data['surf2struct'])
            output['struct2surf'] = torch.from_numpy(data['struct2surf'])
            if self.rotation:
                output['batched_pos'] = torch.matmul(torch.from_numpy(data['batched_pos'])  - center, rot)
            else:
                output['batched_pos'] = torch.from_numpy(data['batched_pos'])  - center
            output['batched_seq'] = torch.from_numpy(data['batched_seq'])

        return output

    def gen_rot(self):
        R = torch.FloatTensor(Rotation.random().as_matrix())
        return R





if __name__ == '__main__':

    print()







