import numpy as np
import torch
import os
from tqdm import tqdm
from torchdrug import data
import utils.load_pdb
import json
import warnings
warnings.filterwarnings('ignore')

def get_coords(residues, mode = 'ca'):
    if mode == 'ca':
        coords = np.array([res['CA'].get_coord() for res in residues])
    elif mode == 'com':
        coords = None
    else:
        raise ValueError(f'Coords cannot be generated for mode {mode}')
    return coords


transfer = {}
map = [5,0,15,12,17,16,1,7,9,11,2,13,8,3,10,6,4,14,19,18]
for i in range(20):
    transfer[i] = map[i]

def transfer_id(array, map):
    return np.vectorize(map.get)(array)


id2residue_symbol = {0: "G", 1: "A", 2: "S", 3: "P", 4: "V", 5: "T", 6: "C", 7: "I", 8: "L", 9: "N",
                         10: "D", 11: "Q", 12: "K", 13: "E", 14: "M", 15: "H", 16: "F", 17: "R", 18: "Y", 19: "W", 20: "X"}
residue_symbol2id = {v: k for k, v in id2residue_symbol.items()}

aa = "ACDEFGHIKLMNPQRSTVWYX"
aa_to_id = {}
for i in range(0, 21):
    aa_to_id[aa[i]] = i

from sklearn.preprocessing import normalize
def orientation(pos):
    u = normalize(X=pos[1:,:] - pos[:-1,:], norm='l2', axis=1)
    u1 = u[1:,:]
    u2 = u[:-1, :]
    b = normalize(X=u2 - u1, norm='l2', axis=1)
    n = normalize(X=np.cross(u2, u1), norm='l2', axis=1)
    o = normalize(X=np.cross(b, n), norm='l2', axis=1)
    ori = np.stack([b, n, o], axis=1)
    return np.concatenate([np.expand_dims(ori[0], 0), ori, np.expand_dims(ori[-1], 0)], axis=0)




def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Data_Pre_Process')
    parser.add_argument('--dataset', default='alf', type=str,
                        choices=['func', 'ec', 'pdbbind', 'fold', 'go', 'rcsb', 'alf'])
    parser.add_argument('--data_dir', default='/media/ymz/807a6847-34c1-4630-bb53-2a277f79be0c/ProteinShake/Swissprot/swissprot_pdb_v4', type=str,
                        metavar='N', help='data root directory')
    parser.add_argument('--pdb_fix', default=False, type=bool)
    parser.add_argument('--surface', default=False, type=bool)
    parser.add_argument('--sequence', default=False, type=bool)

    args = parser.parse_args()
    return args


def main(args, split):

    # not_match = []
    # match= []
    if args.dataset == 'func':
        root = args.data_dir
        output_files = os.path.join(root, 'structure')
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

            pdb = utils.load_pdb.Protein_New.from_pdb(pdb_root, atom_feature=None, bond_feature=None, residue_feature=None)
            if pdb is None:
                print()
                pdb = utils.load_pdb.Protein_New.from_pdb(pdb_root, atom_feature=None, bond_feature=None,
                                                          residue_feature=None)


            idx = pdb.atom_name == pdb.atom_name2id['CA']
            coords = pdb.node_position[idx].numpy()

            idx = pdb.atom2residue[idx]
            amino_ids = pdb.residue_type[idx].numpy()


            # coords_fhh = np.load(os.path.join('/media/ymz/807a6847-34c1-4630-bb53-2a277f79be0c/CDConv/func/coordinates/training', protein_name.replace('_','.') + '.npy'))
            # if coords_fhh.shape != coords.shape:
            #     not_match.append((coords_fhh.shape, coords.shape))
            # else:
            #     match.append(((coords_fhh - coords)**2).mean())


            if args.surface:
                pass

            if args.sequence:
                pass

            np.savez(
                processed_file,
                amino_ids = amino_ids,
                coords = coords
            )

    elif args.dataset == 'pdbbind':

        root = args.data_dir
        if not args.pdb_fix:
            output_files = os.path.join(root, 'structure')
        else:
            output_files = os.path.join(root, 'structure_fixed')
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
            processed_file = os.path.join(output_file, protein_name + '.npz')

            pdb = utils.load_pdb.Protein_New.from_pdb(pdb_root, atom_feature=None, bond_feature=None,
                                                      residue_feature=None)
            if pdb is None:
                print(protein_name + ' could not processed')
                continue
                # 3mv0, 2r75, 1c4u, 3n9r, 3muz
                # pdb = utils.load_pdb.Protein_New.from_pdb(pdb_root, atom_feature=None, bond_feature=None,
                #                                           residue_feature=None)

            idx = pdb.atom_name == pdb.atom_name2id['CA']
            coords = pdb.node_position[idx].numpy()

            idx = pdb.atom2residue[idx]
            amino_ids = pdb.residue_type[idx].numpy()

            if args.surface:
                pass

            if args.sequence:
                pass

            np.savez(
                processed_file,
                amino_ids=amino_ids,
                coords=coords
            )

    elif args.dataset == 'ec':
        root = args.data_dir
        output_files = os.path.join(root, 'structure')
        pdb_files = os.path.join(root, 'EnzymeCommission', split)

        fasta_file = os.path.join(root, split + '.fasta')

        protein_seqs = []
        with open(fasta_file, 'r') as f:
            protein_names = []
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                    # protein_names.append(protein_name.replace('.', '_'))
                else:
                    amino_chain = line.rstrip()
                    amino_ids = []
                    for amino in amino_chain:
                        amino_ids.append(aa_to_id[amino])
                    protein_seqs.append((protein_name, np.array(amino_ids)))

        raw_names = os.listdir(pdb_files)

        for protein_name, amino_ids_cdc in tqdm(protein_seqs):
            output_file = os.path.join(output_files, protein_name)
            os.makedirs(output_file, exist_ok=True)
            processed_file = os.path.join(output_file, protein_name + '.npz')

            coords_cdc = np.load(os.path.join('/media/ymz/807a6847-34c1-4630-bb53-2a277f79be0c/CDConv/ec/coordinates', protein_name + '.npy'))

            if args.surface:
                pass

            if args.sequence:
                pass

            np.savez(
                processed_file,
                amino_ids=amino_ids_cdc,
                coords=coords_cdc.astype(np.float32)
            )

    elif args.dataset == 'fold':
        root = args.data_dir
        output_files = os.path.join(root, 'structure')

        fasta_file = os.path.join(root, split + '.fasta')

        protein_seqs = []
        with open(fasta_file, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    amino_chain = line.rstrip()
                    amino_ids = []
                    for amino in amino_chain:
                        amino_ids.append(aa_to_id[amino])
                    protein_seqs.append((protein_name, np.array(amino_ids)))


        for protein_name, amino_ids_cdc in tqdm(protein_seqs):
            output_file = os.path.join(output_files, protein_name)
            os.makedirs(output_file, exist_ok=True)
            processed_file = os.path.join(output_file, protein_name + '.npz')

            coords_cdc = np.load(os.path.join('/media/ymz/807a6847-34c1-4630-bb53-2a277f79be0c/CDConv/fold/coordinates', split, protein_name + '.npy'))

            if args.surface:
                pass

            if args.sequence:
                pass

            assert len(amino_ids_cdc) > 1 and len(coords_cdc) > 1 and len(amino_ids_cdc) == len(coords_cdc)

            np.savez(
                processed_file,
                amino_ids=amino_ids_cdc,
                coords=coords_cdc
            )

    elif args.dataset == 'go':
        root = args.data_dir
        output_files = os.path.join(root, 'structure')

        fasta_file = os.path.join(root, split + '.fasta')

        protein_seqs = []
        with open(fasta_file, 'r') as f:
            protein_names = []
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                    # protein_names.append(protein_name.replace('.', '_'))
                else:
                    amino_chain = line.rstrip()
                    amino_ids = []
                    for amino in amino_chain:
                        amino_ids.append(aa_to_id[amino])
                    protein_seqs.append((protein_name, np.array(amino_ids)))


        for protein_name, amino_ids_cdc in tqdm(protein_seqs):
            output_file = os.path.join(output_files, protein_name)
            os.makedirs(output_file, exist_ok=True)
            processed_file = os.path.join(output_file, protein_name + '.npz')

            coords_cdc = np.load(os.path.join('/media/ymz/807a6847-34c1-4630-bb53-2a277f79be0c/CDConv/go/coordinates', protein_name + '.npy'))

            if args.surface:
                pass

            if args.sequence:
                pass

            np.savez(
                processed_file,
                amino_ids=amino_ids_cdc,
                coords=coords_cdc.astype(np.float32)
            )

    elif args.dataset == 'rcsb':
        root = args.data_dir
        output_files = '/media/ymz/807a6847-34c1-4630-bb53-2a277f79be0c/UniDrug/Pretraining/raw/RCSB_NPZ'
        pdb_files = os.listdir(root)

        count = 0
        for protein_name in tqdm(pdb_files):
            if '.pdb' not in protein_name:
                continue
            try:
                pdb_file = os.path.join(root, protein_name)

                processed_file = os.path.join(output_files, protein_name.replace('.pdb', '.npz'))

                pdb = utils.load_pdb.Protein_New.from_pdb(pdb_file, atom_feature=None, bond_feature=None,
                                                          residue_feature=None)
                if pdb is None:
                    print()
                    pdb = utils.load_pdb.Protein_New.from_pdb(pdb_file, atom_feature=None, bond_feature=None,
                                                              residue_feature=None)

                idx = pdb.atom_name == pdb.atom_name2id['CA']
                coords = pdb.node_position[idx].numpy()

                idx = pdb.atom2residue[idx]
                amino_ids = pdb.residue_type[idx].numpy()

                pos = coords
                center = np.sum(a=pos, axis=0, keepdims=True) / pos.shape[0]
                pos = pos - center

                ori = orientation(pos)

                np.savez(
                    processed_file,
                    amino_ids=amino_ids,
                    pos=pos.astype(dtype=np.float32),
                    ori=ori.astype(dtype=np.float32)
                )

            except Exception as e:
                count += 1
                print('error_{}'.format(count), protein_name, e)
                continue

    elif args.dataset == 'alf':
        root = args.data_dir
        output_files = '/media/ymz/807a6847-34c1-4630-bb53-2a277f79be0c/UniDrug/Pretraining/raw/SwissProt_NPZ'
        pdb_files = os.listdir(root)

        count = 0
        for protein_name in tqdm(pdb_files):
            if '.pdb' not in protein_name:
                continue
            try:

                pdb_file = '/media/ymz/807a6847-34c1-4630-bb53-2a277f79be0c/ProteinF3S/func/pdb_files/5xjx_K/5xjx_K.pdb'

                processed_file = os.path.join(output_files, protein_name.replace('.pdb', '.npz'))

                pdb = utils.load_pdb.Protein_New.from_pdb(pdb_file, atom_feature=None, bond_feature=None,
                                                          residue_feature=None)
                if pdb is None:
                    print()
                    pdb = utils.load_pdb.Protein_New.from_pdb(pdb_file, atom_feature=None, bond_feature=None,
                                                              residue_feature=None)

                idx = pdb.atom_name == pdb.atom_name2id['CA']
                coords = pdb.node_position[idx].numpy()

                idx = pdb.atom2residue[idx]
                amino_ids = pdb.residue_type[idx].numpy()

                pos = coords
                center = np.sum(a=pos, axis=0, keepdims=True) / pos.shape[0]
                pos = pos - center

                ori = orientation(pos)

                # amino_ids = transfer_id(amino_ids, transfer)

                np.savez(
                    processed_file,
                    amino_ids=amino_ids,
                    pos=pos.astype(dtype=np.float32),
                    ori=ori.astype(dtype=np.float32)
                )

            except Exception as e:
                count += 1
                print('error_{}'.format(count), protein_name, e)
                continue

    print('processed ' + split + ' set')


if __name__ == '__main__':
    args = parse_args()
    # for split in ['training', 'validation', 'testing']:
    # for split in ['train', 'valid', 'test']:
    # for split in ['training', 'validation', 'test_fold', 'test_family', 'test_superfamily']:
    #     main(args, split)
    main(args, None)
    print('Finish')


