import numpy as np
import open3d as o3d
import os



root = '/media/ymz/807a6847-34c1-4630-bb53-2a277f79be0c/ProteinF3S/ec/structure'
protein_names = os.listdir(root)

for i in range(0, len(protein_names), 100):
    name = protein_names[i]

    data_our = np.load(os.path.join(root, name, name + '.npz'))
    pos_our = data_our['coords']

    data_cdc = np.load(os.path.join('/media/ymz/807a6847-34c1-4630-bb53-2a277f79be0c/CDConv/ec/coordinates', name + '.npy'))
    pos_cdc = data_cdc

    data_surf = np.load(os.path.join('/media/ymz/807a6847-34c1-4630-bb53-2a277f79be0c/ProteinF3S/ec/surface', name, name + '.npz'))
    surf = data_surf['xyz']
    atom = data_surf['atom']

    our = o3d.geometry.PointCloud()
    cdc = o3d.geometry.PointCloud()

    pcd_surf = o3d.geometry.PointCloud()
    pcd_atom = o3d.geometry.PointCloud()

    our.points = o3d.utility.Vector3dVector(pos_our)
    cdc.points = o3d.utility.Vector3dVector(pos_cdc + 0.1)

    pcd_surf.points = o3d.utility.Vector3dVector(surf)
    pcd_atom.points = o3d.utility.Vector3dVector(atom - 0.1)

    our.paint_uniform_color([1,0,0])
    cdc.paint_uniform_color([0,0,1])

    pcd_surf.paint_uniform_color([0,1,0])
    pcd_atom.paint_uniform_color([0.3, 0.7, 0.5])

    o3d.visualization.draw_geometries([our, cdc, pcd_atom, pcd_surf])



