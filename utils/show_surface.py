# building surface using dmasif
import os
import numpy as np
from tqdm import tqdm
from utils.helper import *
from utils.Arguments import parser
from utils.geometry_processing import (
    curvatures,
    atoms_to_points_normals,
)


def construct_surface(pdb_root, args):
    data = np.load(pdb_root)
    atom = data['atoms']
    type = data['types']

    atom = torch.from_numpy(atom).cuda()
    atom_type = torch.from_numpy(type).cuda()
    atom_batch = torch.zeros_like(atom)[:, 0]
    atom_batch = atom_batch.int()

    xyz, normal, batch = atoms_to_points_normals(
        atom,
        atom_batch,
        atomtypes=atom_type,
        resolution=args.resolution,
        sup_sampling=args.sup_sampling,
        distance=args.distance,
    )

    P_curvatures = curvatures(
        xyz,
        triangles=None if args.use_mesh else None,
        normals=None if args.use_mesh else normal,
        scales=args.curvature_scales,
        batch=batch,
    )

    return xyz.cpu().numpy().astype('float32'), \
           normal.cpu().numpy().astype('float32'), \
           P_curvatures.cpu().numpy().astype('float32'), \
           atom.cpu().numpy().astype('float32'),\
           atom_type.cpu().numpy().astype('float32'),




if __name__ == "__main__":
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    pdb_root = '/media/ymz/807a6847-34c1-4630-bb53-2a277f79be0c/ProteinF3S/func/pdb_files/2gw2_A/2gw2_A.pdb'

    print("Begin process")

    xyz, normal, curvature, atom, type = construct_surface(pdb_root.replace('pdb_files', 'atom').replace('.pdb', '.npz'))

    import open3d as o3d
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(atom)
    pcd0.paint_uniform_color([1, 0, 0])

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(xyz)
    pcd1.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([pcd0, pcd1])

    print('Finish')





