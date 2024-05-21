import time

import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
import os
import pickle
import torch
import numpy as np
import open3d as o3d
from functools import partial
from torch.utils.data import DataLoader
from torch_scatter import scatter_max, scatter_min, scatter_mean, scatter_sum
from torch_geometric.data import Batch


def calibrate_neighbors(dataset, config, collate_fn, keep_ratio=0.8, samples_threshold=2000):

    if not config.surface:
        return None

    # From config parameter, compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (config.deform_radius + 1) ** 3))
    neighb_hists = np.zeros((config.num_layers, hist_n), dtype=np.int32)

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        batched_input = collate_fn([dataset[i]], config, neighborhood_limits=[hist_n] * 5)

        # update histogram
        counts = [torch.sum(neighb_mat < neighb_mat.shape[0], dim=1).numpy() for neighb_mat in batched_input['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighb_hists += np.vstack(hists)

        # if timer.total_time - last_display > 0.1:
        #     last_display = timer.total_time
        #     print(f"Calib Neighbors {i:08d}: timings {timer.total_time:4.2f}s")

        if np.min(np.sum(neighb_hists, axis=1)) > samples_threshold:
            break

    cumsum = np.cumsum(neighb_hists.T, axis=0)
    percentiles = np.sum(cumsum < (keep_ratio * cumsum[hist_n - 1, :]), axis=0)

    neighborhood_limits = percentiles
    print('\n')

    return neighborhood_limits


def batch_grid_subsampling_kpconv(points, batches_len, features=None, labels=None, sampleDl=0.1, max_p=0, verbose=0, random_grid_orient=True):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    """
    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(points,
                                                          batches_len,
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len)

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points,
                                                                      batches_len,
                                                                      features=features,
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features)

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points,
                                                                    batches_len,
                                                                    classes=labels,
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_labels)

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points,
                                                                              batches_len,
                                                                              features=features,
                                                                              classes=labels,
                                                                              sampleDl=sampleDl,
                                                                              max_p=max_p,
                                                                              verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features), torch.from_numpy(s_labels)


def batch_neighbors_kpconv(queries, supports, q_batches, s_batches, radius, max_neighbors):
    """
    Computes neighbors for a batch of queries and supports, apply radius search
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """

    neighbors = cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)
    if max_neighbors > 0:
        return torch.from_numpy(neighbors[:, :max_neighbors])
    else:
        return torch.from_numpy(neighbors)


def get_dataloader(dataset, split, cfg, neighborhood_limits=None):
    """
    Builds the dataset loader (including getting the dataset).
    """
    if cfg.use_superpoint:
        if cfg.dataset == 'pdbbind':
            collate_fn = collate_fn_hybrid
        elif cfg.dataset in ['func', 'ec', 'go']:
           collate_fn = collate_fn_hybrid
    else:
        collate_fn = collate_fn_hybrid_dense

    shuffle = split == "training"
    batch_size = cfg.batch_size
    num_workers = cfg.workers

    if neighborhood_limits is None:
        neighborhood_limits = calibrate_neighbors(dataset, cfg, collate_fn=collate_fn)
    if cfg.surface:
        print("neighborhood:", neighborhood_limits)

    if cfg.acc_iter > 1:
        if split == "training":
            batch_size = batch_size // cfg.acc_iter

    loader = DataLoader(
        dataset=dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(collate_fn, config=cfg, neighborhood_limits=neighborhood_limits),
        drop_last=False
    )

    return loader, neighborhood_limits


# def collate_fn_surface(list_data, config, neighborhood_limits):
#     batched_points_list = []
#     batched_geos_list = []
#     batched_chems_list = []
#     batched_features_list = []
#     batched_lengths_list = []
#     batch_labels_list = []
#
#     for ind, batch_i in enumerate(list_data):
#         point = batch_i['point']
#         geo = batch_i['geo']
#         chem = batch_i['chem']
#         label = batch_i['label']
#
#         batched_points_list.append(point)
#         batched_geos_list.append(geo)
#         batched_chems_list.append(chem)
#         batched_features_list.append(torch.ones(len(point), 1))
#         batched_lengths_list.append(len(point))
#         batch_labels_list.append(label)
#
#
#     batched_points = torch.cat(batched_points_list)
#     batched_geos_list = torch.cat(batched_geos_list)
#     batched_chems_list = torch.cat(batched_chems_list)
#     batched_features = torch.cat(batched_features_list)
#     batched_lengths = torch.tensor(batched_lengths_list).int()
#     batch_labels_list = torch.cat(batch_labels_list)
#
#
#     # Starting radius of convolutions
#     r_normal = config.first_subsampling_dl * config.conv_radius
#
#     # Starting layer
#     layer_blocks = []
#     layer = 0
#
#     # Lists of inputs
#     input_points = []
#     input_neighbors = []
#     input_pools = []
#     # input_upsamples = []
#     input_batches_len = []
#
#
#     # construt kpfcn inds
#     for block_i, block in enumerate(config.architectures):
#
#         # # Stop when meeting a global pooling or upsampling
#         # if 'global' in block or 'upsample' in block:
#         #     break
#
#         # Get all blocks of the layer
#         if not ('pool' in block or 'strided' in block):
#             layer_blocks += [block]
#             if block_i < len(config.architectures) - 1 and not ('upsample' in config.architectures[block_i + 1]):
#                 continue
#
#         # Convolution neighbors indices
#         # *****************************
#
#         if layer_blocks:
#             # Convolutions are done in this layer, compute the neighbors with the good radius
#             if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
#                 r = r_normal * config.deform_radius / config.conv_radius
#             else:
#                 r = r_normal
#             conv_i = batch_neighbors_kpconv(batched_points, batched_points, batched_lengths, batched_lengths, r,
#                                             neighborhood_limits[layer])
#         else:
#             # This layer only perform pooling, no neighbors required
#             conv_i = torch.zeros((0, 1), dtype=torch.int64)
#
#         # Pooling neighbors indices
#         # *************************
#
#         # If end of layer is a pooling operation
#         if 'pool' in block or 'strided' in block:
#
#             # New subsampling length
#             dl = 2 * r_normal / config.conv_radius
#
#             # Subsampled points
#             pool_p, pool_b = batch_grid_subsampling_kpconv(batched_points, batched_lengths, sampleDl=dl)
#
#             # Radius of pooled neighbors
#             if 'deformable' in block:
#                 r = r_normal * config.deform_radius / config.conv_radius
#             else:
#                 r = r_normal
#
#             # Subsample indices
#             pool_i = batch_neighbors_kpconv(pool_p, batched_points, pool_b, batched_lengths, r,
#                                             neighborhood_limits[layer])
#
#             # # Upsample indices (with the radius of the next layer to keep wanted density)
#             # up_i = batch_neighbors_kpconv(batched_points, pool_p, batched_lengths, pool_b, 2 * r,
#             #                               neighborhood_limits[layer])
#
#
#         else:
#             # No pooling in the end of this layer, no pooling indices required
#             pool_i = torch.zeros((0, 1), dtype=torch.int64)
#             pool_p = torch.zeros((0, 3), dtype=torch.float32)
#             pool_b = torch.zeros((0,), dtype=torch.int64)
#             # up_i = torch.zeros((0, 1), dtype=torch.int64)
#
#         # Updating input lists
#         input_points += [batched_points.float()]
#         input_neighbors += [conv_i.long()]
#         input_pools += [pool_i.long()]
#         # input_upsamples += [up_i.long()]
#         input_batches_len += [batched_lengths]
#
#         # New points for next layer
#         batched_points = pool_p
#         batched_lengths = pool_b
#
#         # Update radius and reset blocks
#         r_normal *= 2
#         layer += 1
#         layer_blocks = []
#
#
#     dict_inputs = {
#         'points': input_points,
#         'neighbors': input_neighbors,
#         'pools': input_pools,
#         # 'upsamples': input_upsamples,
#         'features': batched_features.float(),
#         'geo': batched_geos_list,
#         'chem': batched_chems_list,
#         'stack_lengths': input_batches_len,
#         'label': batch_labels_list
#     }
#
#     return dict_inputs




# # old
#
# def collate_fn_hybrid(list_data, config, neighborhood_limits):
#     dict_inputs = {}
#     batch_labels_list = []
#     batch_ligands_list = []
#
#     if config.surface:
#         batched_points_list = []
#         batched_geos_list = []
#         batched_chems_list = []
#         batched_features_list = []
#         batched_lengths_list = []
#     if config.sequence:
#         batched_amino_sequence = []
#     if config.structure:
#         batched_x = []
#         batched_ori = []
#         batched_seq = []
#         batched_pos = []
#         batched_index = []
#
#     if config.fusion_type == 'msf':
#         surf2struct_offline_list = []
#         struct2surf_offline_list = []
#         batched_pos_offline_list = []
#         batched_seq_offline_list = []
#         batched_index_offline_list = []
#
#     length_surf = 0
#     length_struct = 0
#     for ind, batch_i in enumerate(list_data):
#         label = batch_i['label']
#         batch_labels_list.append(label)
#
#         if config.surface:
#             point = batch_i['point']
#             geo = batch_i['geo']
#             chem = batch_i['chem']
#
#             batched_points_list.append(point)
#             batched_geos_list.append(geo)
#             batched_chems_list.append(chem)
#             batched_features_list.append(torch.ones(len(point), 1))
#             batched_lengths_list.append(len(point))
#
#         if config.sequence:
#             amino_sequence = batch_i['sequence']
#             batched_amino_sequence.append(amino_sequence)
#
#         if config.structure:
#             batched_x.append(batch_i['x'])
#             batched_ori.append(batch_i['ori'])
#             batched_seq.append(batch_i['seq'])
#             batched_pos.append(batch_i['pos'])
#             batched_index.append((ind * torch.ones(len(batch_i['x']), dtype=torch.int64)))
#
#         if config.fusion_type:
#             surf2struct_offline_list.append(batch_i['surf2struct'] + length_surf)
#             struct2surf_offline_list.append(batch_i['struct2surf'] + length_struct)
#             batched_pos_offline_list.append(batch_i['batched_pos'])
#             batched_seq_offline_list.append(batch_i['batched_seq'])
#             batched_index_offline_list.append(ind * torch.ones((len(batch_i['x']) + 1)//2, dtype=torch.int64))
#             length_surf += len(point)
#             length_struct += len(batch_i['x'])
#
#         if config.dataset == 'pdbbind':
#             batch_ligands_list.append(batch_i['ligand'])
#
#
#     batch_labels_list = torch.cat(batch_labels_list)
#     dict_inputs['label'] = batch_labels_list
#
#     if config.fusion_type:
#         surf2struct_offline_list = torch.cat(surf2struct_offline_list, dim=0)
#         struct2surf_offline_list = torch.cat(struct2surf_offline_list, dim=0)
#         batched_pos_offline_list = torch.cat(batched_pos_offline_list, dim=0)
#         batched_seq_offline_list = torch.cat(batched_seq_offline_list, dim=0)
#         batched_index_offline_list = torch.cat(batched_index_offline_list, dim=0)
#
#
#     if config.sequence:
#         _, batch_strs, batch_tokens = config.batch_converter(batched_amino_sequence)
#         batch_lens = (batch_tokens != config.alphabet.padding_idx).sum(1)
#         batch_tokens = batch_tokens
#         dict_inputs['tokens'] = batch_tokens
#         dict_inputs['token_lens'] = batch_lens
#
#     if config.structure:
#         batched_x = torch.cat(batched_x, dim=0)
#         batched_ori = torch.cat(batched_ori, dim=0)
#         batched_seq = torch.cat(batched_seq, dim=0)
#         batched_pos = torch.cat(batched_pos, dim=0)
#         batched_index = torch.cat(batched_index, dim=0)
#
#         dict_inputs['x'] = batched_x
#         dict_inputs['ori'] = batched_ori
#         dict_inputs['seq'] = batched_seq
#         dict_inputs['pos'] = batched_pos
#         dict_inputs['batch'] = batched_index
#
#     if config.surface:
#         batched_points = torch.cat(batched_points_list)
#         batched_geos_list = torch.cat(batched_geos_list)
#         batched_chems_list = torch.cat(batched_chems_list)
#         batched_features = torch.cat(batched_features_list)
#         batched_lengths = torch.tensor(batched_lengths_list).int()
#
#
#         # Starting radius of convolutions
#         r_normal = config.first_subsampling_dl * config.conv_radius
#
#         # Starting layer
#         layer_blocks = []
#         layer = 0
#
#         # Lists of inputs
#         input_points = []
#         input_neighbors = []
#         input_pools = []
#         # input_upsamples = []
#         input_batches_len = []
#
#         surf2struct_list = []
#         struct2surf_list = []
#
#         # construt kpfcn inds
#         for block_i, block in enumerate(config.architectures):
#
#             # # Stop when meeting a global pooling or upsampling
#             # if 'global' in block or 'upsample' in block:
#             #     break
#
#             # Get all blocks of the layer
#             if not ('pool' in block or 'strided' in block):
#                 layer_blocks += [block]
#                 if block_i < len(config.architectures) - 1 and not ('upsample' in config.architectures[block_i + 1]):
#                     continue
#
#             # Convolution neighbors indices
#             # *****************************
#
#             if layer_blocks:
#                 # Convolutions are done in this layer, compute the neighbors with the good radius
#                 if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
#                     r = r_normal * config.deform_radius / config.conv_radius
#                 else:
#                     r = r_normal
#                 conv_i = batch_neighbors_kpconv(batched_points, batched_points, batched_lengths, batched_lengths, r,
#                                                 neighborhood_limits[layer])
#
#                 if config.fusion_type == 'cascade' and block_i == 10:
#                     batched_lengths_struct = []
#                     for iii in range(batched_index.max() + 1):
#                         batched_lengths_struct.append( (batched_index == iii).sum())
#                     batched_lengths_struct = torch.tensor(batched_lengths_struct).int()
#
#                     if config.surface2struct:
#                         surf2struct = batch_neighbors_kpconv(batched_pos, batched_points, batched_lengths_struct, batched_lengths, 10*r, 1).squeeze().long()
#
#                 if config.fusion_type == 'msf':
#                     batched_lengths_struct = []
#                     for iii in range(batched_index.max() + 1):
#                         batched_lengths_struct.append((batched_index == iii).sum())
#                     batched_lengths_struct = torch.tensor(batched_lengths_struct).int()
#
#                     if config.bidirectional:
#                         if block_i == 2:
#                             surf2struct_list.append(surf2struct_offline_list)
#                             struct2surf_list.append(struct2surf_offline_list)
#
#                             batched_pos = batched_pos_offline_list
#                             batched_seq= batched_seq_offline_list
#                             batched_index = batched_index_offline_list
#
#                             # batched_pos, batched_seq, batched_index = downsample_struct(batched_pos, batched_seq,
#                             #                                                             batched_index)
#
#                         else:
#                             surf2struct_list.append(batch_neighbors_kpconv(batched_pos, batched_points, batched_lengths_struct, batched_lengths,
#                                                                             config.scale * r, config.K_surf2struct).long())
#
#                             struct2surf_list.append(batch_neighbors_kpconv(batched_points, batched_pos, batched_lengths, batched_lengths_struct,
#                                                                             config.scale * r, config.K_struct2surf).long())
#                             batched_pos, batched_seq, batched_index = downsample_struct(batched_pos, batched_seq, batched_index)
#
#
#                     # else:
#                     #     if config.surface2struct:
#                     #         surf2struct_list.append()
#                     #     else:
#                     #         struct2surf_list.append()
#
#             else:
#                 # This layer only perform pooling, no neighbors required
#                 conv_i = torch.zeros((0, 1), dtype=torch.int64)
#
#             # Pooling neighbors indices
#             # *************************
#
#             # If end of layer is a pooling operation
#             if 'pool' in block or 'strided' in block:
#
#                 # New subsampling length
#                 dl = 2 * r_normal / config.conv_radius
#
#                 # Subsampled points
#                 pool_p, pool_b = batch_grid_subsampling_kpconv(batched_points, batched_lengths, sampleDl=dl)
#
#                 # Radius of pooled neighbors
#                 if 'deformable' in block:
#                     r = r_normal * config.deform_radius / config.conv_radius
#                 else:
#                     r = r_normal
#
#                 # Subsample indices
#                 pool_i = batch_neighbors_kpconv(pool_p, batched_points, pool_b, batched_lengths, r,
#                                                 neighborhood_limits[layer])
#
#                 # # Upsample indices (with the radius of the next layer to keep wanted density)
#                 # up_i = batch_neighbors_kpconv(batched_points, pool_p, batched_lengths, pool_b, 2 * r,
#                 #                               neighborhood_limits[layer])
#
#
#             else:
#                 # No pooling in the end of this layer, no pooling indices required
#                 pool_i = torch.zeros((0, 1), dtype=torch.int64)
#                 pool_p = torch.zeros((0, 3), dtype=torch.float32)
#                 pool_b = torch.zeros((0,), dtype=torch.int64)
#                 # up_i = torch.zeros((0, 1), dtype=torch.int64)
#
#             # Updating input lists
#             input_points += [batched_points.float()]
#             input_neighbors += [conv_i.long()]
#             input_pools += [pool_i.long()]
#             # input_upsamples += [up_i.long()]
#             input_batches_len += [batched_lengths]
#
#             # New points for next layer
#             batched_points = pool_p
#             batched_lengths = pool_b
#
#             # Update radius and reset blocks
#             r_normal *= 2
#             layer += 1
#             layer_blocks = []
#
#         dict_inputs['points'] = input_points
#         dict_inputs['neighbors'] = input_neighbors
#         dict_inputs['pools'] = input_pools
#         dict_inputs['features'] = batched_features.float()
#         dict_inputs['geo'] = batched_geos_list
#         dict_inputs['chem'] = batched_chems_list
#         dict_inputs['stack_lengths'] = input_batches_len
#
#         if config.fusion_type == 'cascade' and config.surface2struct and config.use_superpoint:
#             dict_inputs['surf2struct'] = surf2struct
#         if config.fusion_type == 'msf':
#             dict_inputs['surf2struct_list'] = surf2struct_list
#             dict_inputs['struct2surf_list'] = struct2surf_list
#
#     if config.dataset == 'pdbbind':
#         [delattr(data, 'mess_idx') for data in batch_ligands_list
#          if hasattr(data, 'mess_idx')]
#         dict_inputs['ligand'] = Batch.from_data_list(batch_ligands_list)
#
#     return dict_inputs

def downsample_struct(pos, seq, batch):
    idx = torch.div(seq.squeeze(1), 2, rounding_mode='floor')
    idx = torch.cat([idx, idx[-1].view((1,))])

    idx = (idx[0:-1] != idx[1:]).to(torch.float32)
    idx = torch.cumsum(idx, dim=0) - idx
    idx = idx.to(torch.int64)
    pos = scatter_mean(src=pos, index=idx, dim=0)
    seq = scatter_max(src=torch.div(seq, 2, rounding_mode='floor'), index=idx, dim=0)[0]
    batch = scatter_max(src=batch, index=idx, dim=0)[0]

    return pos, seq, batch


def collate_fn_hybrid_dense(list_data, config, neighborhood_limits):
    dict_inputs = {}
    batch_labels_list = []

    if config.surface:
        batched_points_list = []
        batched_geos_list = []
        batched_chems_list = []
        batched_features_list = []
        batched_lengths_list = []
    if config.sequence:
        batched_amino_sequence = []
    if config.structure:
        batched_x = []
        batched_ori = []
        batched_seq = []
        batched_pos = []
        batched_index = []

    for ind, batch_i in enumerate(list_data):
        label = batch_i['label']
        batch_labels_list.append(label)

        if config.surface:
            point = batch_i['point']
            geo = batch_i['geo']
            chem = batch_i['chem']

            batched_points_list.append(point)
            batched_geos_list.append(geo)
            batched_chems_list.append(chem)
            batched_features_list.append(torch.ones(len(point), 1))
            batched_lengths_list.append(len(point))

        if config.sequence:
            amino_sequence = batch_i['sequence']
            batched_amino_sequence.append(amino_sequence)

        if config.structure:
            batched_x.append(batch_i['x'])
            batched_ori.append(batch_i['ori'])
            batched_seq.append(batch_i['seq'])
            batched_pos.append(batch_i['pos'])
            batched_index.append((ind * torch.ones(len(batch_i['x']), dtype=torch.int64)))

    batch_labels_list = torch.cat(batch_labels_list)
    dict_inputs['label'] = batch_labels_list


    if config.sequence:
        _, batch_strs, batch_tokens = config.batch_converter(batched_amino_sequence)
        batch_lens = (batch_tokens != config.alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens
        dict_inputs['tokens'] = batch_tokens
        dict_inputs['token_lens'] = batch_lens

    if config.structure:
        batched_x = torch.cat(batched_x, dim=0)
        batched_ori = torch.cat(batched_ori, dim=0)
        batched_seq = torch.cat(batched_seq, dim=0)
        batched_pos = torch.cat(batched_pos, dim=0)
        batched_index = torch.cat(batched_index, dim=0)

        dict_inputs['x'] = batched_x
        dict_inputs['ori'] = batched_ori
        dict_inputs['seq'] = batched_seq
        dict_inputs['pos'] = batched_pos
        dict_inputs['batch'] = batched_index

    if config.surface:
        batched_points = torch.cat(batched_points_list)
        batched_geos_list = torch.cat(batched_geos_list)
        batched_chems_list = torch.cat(batched_chems_list)
        batched_features = torch.cat(batched_features_list)
        batched_lengths = torch.tensor(batched_lengths_list).int()


        # Starting radius of convolutions
        r_normal = config.first_subsampling_dl * config.conv_radius

        # Starting layer
        layer_blocks = []
        layer = 0

        # Lists of inputs
        input_points = []
        input_neighbors = []
        input_pools = []
        input_upsamples = []
        input_batches_len = []

        # construt kpfcn inds
        for block_i, block in enumerate(config.architectures):

            # Stop when meeting a global pooling or upsampling
            if 'global' in block or 'upsample' in block:
                break

            # Get all blocks of the layer
            if not ('pool' in block or 'strided' in block):
                layer_blocks += [block]
                if block_i < len(config.architectures) - 1 and not ('upsample' in config.architectures[block_i + 1]):
                    continue

            # Convolution neighbors indices
            # *****************************

            if layer_blocks:
                # Convolutions are done in this layer, compute the neighbors with the good radius
                if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
                    r = r_normal * config.deform_radius / config.conv_radius
                else:
                    r = r_normal
                conv_i = batch_neighbors_kpconv(batched_points, batched_points, batched_lengths, batched_lengths, r,
                                                neighborhood_limits[layer])

                if config.fusion_type == 'cascade' and block_i == 2:
                    batched_lengths_struct = []
                    for iii in range(batched_index.max() + 1):
                        batched_lengths_struct.append( (batched_index == iii).sum())
                    batched_lengths_struct = torch.tensor(batched_lengths_struct).int()

                    if config.surface2struct:
                        surf2struct = batch_neighbors_kpconv(batched_pos, batched_points, batched_lengths_struct, batched_lengths, 100*r, 1).squeeze().long()
            else:
                # This layer only perform pooling, no neighbors required
                conv_i = torch.zeros((0, 1), dtype=torch.int64)

            # Pooling neighbors indices
            # *************************

            # If end of layer is a pooling operation
            if 'pool' in block or 'strided' in block:

                # New subsampling length
                dl = 2 * r_normal / config.conv_radius

                # Subsampled points
                pool_p, pool_b = batch_grid_subsampling_kpconv(batched_points, batched_lengths, sampleDl=dl)

                # Radius of pooled neighbors
                if 'deformable' in block:
                    r = r_normal * config.deform_radius / config.conv_radius
                else:
                    r = r_normal

                # Subsample indices
                pool_i = batch_neighbors_kpconv(pool_p, batched_points, pool_b, batched_lengths, r,
                                                neighborhood_limits[layer])

                # Upsample indices (with the radius of the next layer to keep wanted density)
                up_i = batch_neighbors_kpconv(batched_points, pool_p, batched_lengths, pool_b, 2 * r,
                                              neighborhood_limits[layer])


            else:
                # No pooling in the end of this layer, no pooling indices required
                pool_i = torch.zeros((0, 1), dtype=torch.int64)
                pool_p = torch.zeros((0, 3), dtype=torch.float32)
                pool_b = torch.zeros((0,), dtype=torch.int64)
                up_i = torch.zeros((0, 1), dtype=torch.int64)

            # Updating input lists
            input_points += [batched_points.float()]
            input_neighbors += [conv_i.long()]
            input_pools += [pool_i.long()]
            input_upsamples += [up_i.long()]
            input_batches_len += [batched_lengths]

            # New points for next layer
            batched_points = pool_p
            batched_lengths = pool_b

            # Update radius and reset blocks
            r_normal *= 2
            layer += 1
            layer_blocks = []

        dict_inputs['points'] = input_points
        dict_inputs['neighbors'] = input_neighbors
        dict_inputs['pools'] = input_pools
        dict_inputs['features'] = batched_features.float()
        dict_inputs['geo'] = batched_geos_list
        dict_inputs['chem'] = batched_chems_list
        dict_inputs['stack_lengths'] = input_batches_len
        dict_inputs['upsamples'] =  input_upsamples

        if config.fusion_type == 'cascade' and config.surface2struct:
            dict_inputs['surf2struct'] = surf2struct


    return dict_inputs



    # ### debug
    # import open3d as o3d
    # struct = dict_inputs['pos'][dict_inputs['batch'] == 0]
    # surf = dict_inputs['points'][0][:dict_inputs['stack_lengths'][0][0]]
    #
    # pcd_struct = o3d.geometry.PointCloud()
    # pcd_surf = o3d.geometry.PointCloud()
    # pcd_conv = o3d.geometry.PointCloud()
    #
    #
    # pcd_surf1 = o3d.geometry.PointCloud()
    # pcd_surf2 = o3d.geometry.PointCloud()
    # pcd_surf3 = o3d.geometry.PointCloud()
    #
    # pcd_surf.points = o3d.utility.Vector3dVector(surf.numpy())
    #
    #
    # pcd_surf1.points = o3d.utility.Vector3dVector(dict_inputs['points'][1][:dict_inputs['stack_lengths'][1][0]].numpy() + 10)
    # pcd_surf2.points = o3d.utility.Vector3dVector(dict_inputs['points'][2][:dict_inputs['stack_lengths'][2][0]].numpy() + 20)
    # pcd_surf3.points = o3d.utility.Vector3dVector(dict_inputs['points'][3][:dict_inputs['stack_lengths'][3][0]].numpy() + 30)
    #
    #
    # pcd_surf.paint_uniform_color([1, 0, 0])
    # pcd_surf1.paint_uniform_color([0, 1, 0])
    # pcd_surf2.paint_uniform_color([0, 0, 1])
    # pcd_surf3.paint_uniform_color([0,0,0])
    #
    #
    # o3d.visualization.draw_geometries([pcd_struct, pcd_surf])

def collate_fn_hybrid(list_data, config, neighborhood_limits):
    dict_inputs = {}
    batch_labels_list = []
    batch_ligands_list = []

    if config.surface:
        batched_points_list = []
        batched_geos_list = []
        batched_chems_list = []
        batched_features_list = []
        batched_lengths_list = []
    if config.sequence:
        batched_amino_sequence = []
    if config.structure:
        batched_x = []
        batched_ori = []
        batched_seq = []
        batched_pos = []
        batched_index = []

    if config.fusion_type == 'msf':
        surf2struct_offline_list = []
        struct2surf_offline_list = []
        batched_pos_offline_list = []
        batched_seq_offline_list = []
        batched_index_offline_list = []

    length_surf = 0
    length_struct = 0
    for ind, batch_i in enumerate(list_data):
        label = batch_i['label']
        batch_labels_list.append(label)

        if config.surface:
            point = batch_i['point']
            geo = batch_i['geo']
            chem = batch_i['chem']

            batched_points_list.append(point)
            batched_geos_list.append(geo)
            batched_chems_list.append(chem)
            batched_features_list.append(torch.ones(len(point), 1))
            batched_lengths_list.append(len(point))

        if config.sequence:
            amino_sequence = batch_i['sequence']
            batched_amino_sequence.append(amino_sequence)

        if config.structure:
            batched_x.append(batch_i['x'])
            batched_ori.append(batch_i['ori'])
            batched_seq.append(batch_i['seq'])
            batched_pos.append(batch_i['pos'])
            batched_index.append((ind * torch.ones(len(batch_i['x']), dtype=torch.int64)))

        if config.fusion_type:
            surf2struct_offline_list.append(batch_i['surf2struct'] + length_surf)
            struct2surf_offline_list.append(batch_i['struct2surf'] + length_struct)
            batched_pos_offline_list.append(batch_i['batched_pos'])
            batched_seq_offline_list.append(batch_i['batched_seq'])
            batched_index_offline_list.append(ind * torch.ones((len(batch_i['x']) + 1)//2, dtype=torch.int64))
            length_surf += len(point)
            length_struct += len(batch_i['x'])

        if config.dataset == 'pdbbind':
            batch_ligands_list.append(batch_i['ligand'])


    batch_labels_list = torch.cat(batch_labels_list)
    dict_inputs['label'] = batch_labels_list

    if config.fusion_type:
        surf2struct_offline_list = torch.cat(surf2struct_offline_list, dim=0)
        struct2surf_offline_list = torch.cat(struct2surf_offline_list, dim=0)
        batched_pos_offline_list = torch.cat(batched_pos_offline_list, dim=0)
        batched_seq_offline_list = torch.cat(batched_seq_offline_list, dim=0)
        batched_index_offline_list = torch.cat(batched_index_offline_list, dim=0)


    if config.sequence:
        _, batch_strs, batch_tokens = config.batch_converter(batched_amino_sequence)
        batch_lens = (batch_tokens != config.alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens
        dict_inputs['tokens'] = batch_tokens
        dict_inputs['token_lens'] = batch_lens

    if config.structure:
        batched_x = torch.cat(batched_x, dim=0)
        batched_ori = torch.cat(batched_ori, dim=0)
        batched_seq = torch.cat(batched_seq, dim=0)
        batched_pos = torch.cat(batched_pos, dim=0)
        batched_index = torch.cat(batched_index, dim=0)

        dict_inputs['x'] = batched_x
        dict_inputs['ori'] = batched_ori
        dict_inputs['seq'] = batched_seq
        dict_inputs['pos'] = batched_pos
        dict_inputs['batch'] = batched_index

    if config.surface:
        batched_points = torch.cat(batched_points_list)
        batched_geos_list = torch.cat(batched_geos_list)
        batched_chems_list = torch.cat(batched_chems_list)
        batched_features = torch.cat(batched_features_list)
        batched_lengths = torch.tensor(batched_lengths_list).int()


        # Starting radius of convolutions
        r_normal = config.first_subsampling_dl * config.conv_radius

        # Starting layer
        layer_blocks = []
        layer = 0

        # Lists of inputs
        input_points = []
        input_neighbors = []
        input_pools = []
        # input_upsamples = []
        input_batches_len = []

        surf2struct_list = []
        struct2surf_list = []

        # construt kpfcn inds
        for block_i, block in enumerate(config.architectures):

            # # Stop when meeting a global pooling or upsampling
            # if 'global' in block or 'upsample' in block:
            #     break

            # Get all blocks of the layer
            if not ('pool' in block or 'strided' in block):
                layer_blocks += [block]
                if block_i < len(config.architectures) - 1 and not ('upsample' in config.architectures[block_i + 1]):
                    continue

            # Convolution neighbors indices
            # *****************************

            if layer_blocks:
                # Convolutions are done in this layer, compute the neighbors with the good radius
                if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
                    r = r_normal * config.deform_radius / config.conv_radius
                else:
                    r = r_normal
                conv_i = batch_neighbors_kpconv(batched_points, batched_points, batched_lengths, batched_lengths, r,
                                                neighborhood_limits[layer])

                if config.fusion_type == 'cascade' and block_i == 10:
                    batched_lengths_struct = []
                    for iii in range(batched_index.max() + 1):
                        batched_lengths_struct.append( (batched_index == iii).sum())
                    batched_lengths_struct = torch.tensor(batched_lengths_struct).int()

                    if config.surface2struct:
                        surf2struct = batch_neighbors_kpconv(batched_pos, batched_points, batched_lengths_struct, batched_lengths, 10*r, 1).squeeze().long()

                if config.fusion_type == 'msf':
                    batched_lengths_struct = []
                    for iii in range(batched_index.max() + 1):
                        batched_lengths_struct.append((batched_index == iii).sum())
                    batched_lengths_struct = torch.tensor(batched_lengths_struct).int()

                    if config.bidirectional:
                        if block_i == 2:
                            if config.use_dense:
                                surf2struct_list.append(surf2struct_offline_list)
                                struct2surf_list.append(struct2surf_offline_list)
                            else:
                                pass
                            # print()
                            # import open3d as o3d
                            # pcd_struct = o3d.geometry.PointCloud()
                            # pcd_surf = o3d.geometry.PointCloud()
                            #
                            # pcd_struct.points = o3d.utility.Vector3dVector(batched_pos[:batched_lengths_struct[0]].numpy())
                            # pcd_surf.points = o3d.utility.Vector3dVector(batched_points[:batched_lengths[0]].numpy())
                            #
                            # pcd_struct.paint_uniform_color([1, 0 , 0])
                            # pcd_surf.paint_uniform_color([0,1,0])
                            #
                            # o3d.visualization.draw_geometries([pcd_struct, pcd_surf])
                            #
                            # print(batched_lengths_struct, batched_lengths)
                            # print(batched_lengths.sum(), surf2struct_list[-1].max())
                            # print(batched_lengths_struct.sum(), struct2surf_list[-1].max())
                            #
                            #
                            # for kk in range(0,len(batched_pos),len(batched_pos)//10):
                            #
                            #     pcd_surf_part = o3d.geometry.PointCloud()
                            #     pcd_struct_query = o3d.geometry.PointCloud()
                            #
                            #     pcd_struct_query.points = o3d.utility.Vector3dVector(batched_pos[kk].numpy().reshape(1,3))
                            #     pcd_surf_part.points = o3d.utility.Vector3dVector(
                            #         batched_points[surf2struct_list[-1][kk][surf2struct_list[-1][kk]!=batched_lengths.sum()]].numpy() + 0.5)
                            #
                            #     pcd_struct_query.paint_uniform_color([1, 0, 0])
                            #     pcd_surf_part.paint_uniform_color([0, 0, 1])
                            #
                            #     o3d.visualization.draw_geometries([pcd_struct_query, pcd_surf_part, pcd_surf])
                            #
                            #
                            # pcd_inner = o3d.geometry.PointCloud()
                            # pcd_inner.points = o3d.utility.Vector3dVector(batched_pos[(surf2struct_list[-1] == batched_lengths.sum()).sum(-1) == 16].numpy() - 0.5)
                            # pcd_inner.paint_uniform_color([0,0,1])
                            # o3d.visualization.draw_geometries([pcd_inner, pcd_struct, pcd_surf])
                            # o3d.visualization.draw_geometries([pcd_inner, pcd_struct])

                            batched_pos = batched_pos_offline_list
                            batched_seq= batched_seq_offline_list
                            batched_index = batched_index_offline_list

                            # batched_pos, batched_seq, batched_index = downsample_struct(batched_pos, batched_seq,
                            #                                                             batched_index)

                        else:
                            surf2struct_list.append(batch_neighbors_kpconv(batched_pos, batched_points, batched_lengths_struct, batched_lengths,
                                                                            config.scale * r, config.K_surf2struct).long())

                            struct2surf_list.append(batch_neighbors_kpconv(batched_points, batched_pos, batched_lengths, batched_lengths_struct,
                                                                            config.scale * r, config.K_struct2surf).long())
                            # print()
                            # import open3d as o3d
                            # pcd_struct = o3d.geometry.PointCloud()
                            # pcd_surf = o3d.geometry.PointCloud()
                            #
                            # pcd_struct.points = o3d.utility.Vector3dVector(batched_pos[:batched_lengths_struct[0]].numpy())
                            # pcd_surf.points = o3d.utility.Vector3dVector(batched_points[:batched_lengths[0]].numpy())
                            #
                            # pcd_struct.paint_uniform_color([1, 0 , 0])
                            # pcd_surf.paint_uniform_color([0,1,0])
                            #
                            # o3d.visualization.draw_geometries([pcd_struct, pcd_surf])
                            #
                            # print(batched_lengths_struct, batched_lengths)
                            # print(batched_lengths.sum(), surf2struct_list[-1].max())
                            # print(batched_lengths_struct.sum(), struct2surf_list[-1].max())
                            #
                            #
                            # for kk in range(0,len(batched_pos),len(batched_pos)//10):
                            #
                            #     pcd_surf_part = o3d.geometry.PointCloud()
                            #     pcd_struct_query = o3d.geometry.PointCloud()
                            #
                            #     pcd_struct_query.points = o3d.utility.Vector3dVector(batched_pos[kk].numpy().reshape(1,3))
                            #     pcd_surf_part.points = o3d.utility.Vector3dVector(
                            #         batched_points[surf2struct_list[-1][kk][surf2struct_list[-1][kk]!=batched_lengths.sum()]].numpy() + 0.5)
                            #
                            #     pcd_struct_query.paint_uniform_color([1, 0, 0])
                            #     pcd_surf_part.paint_uniform_color([0, 0, 1])
                            #
                            #     o3d.visualization.draw_geometries([pcd_struct_query, pcd_surf_part, pcd_surf])
                            #
                            #
                            # pcd_inner = o3d.geometry.PointCloud()
                            # pcd_inner.points = o3d.utility.Vector3dVector(batched_pos[(surf2struct_list[-1] == batched_lengths.sum()).sum(-1) == 16].numpy() - 0.5)
                            # pcd_inner.paint_uniform_color([0,0,1])
                            # o3d.visualization.draw_geometries([pcd_inner, pcd_struct, pcd_surf])
                            # o3d.visualization.draw_geometries([pcd_inner, pcd_struct])

                            batched_pos, batched_seq, batched_index = downsample_struct(batched_pos, batched_seq, batched_index)


            else:
                # This layer only perform pooling, no neighbors required
                conv_i = torch.zeros((0, 1), dtype=torch.int64)

            # Pooling neighbors indices
            # *************************

            # If end of layer is a pooling operation
            if 'pool' in block or 'strided' in block:

                # New subsampling length
                dl =  r_normal / config.conv_radius + config.first_subsampling_dl

                # Subsampled points
                pool_p, pool_b = batch_grid_subsampling_kpconv(batched_points, batched_lengths, sampleDl=dl)

                # Radius of pooled neighbors
                if 'deformable' in block:
                    r = r_normal * config.deform_radius / config.conv_radius
                else:
                    r = r_normal

                # Subsample indices
                pool_i = batch_neighbors_kpconv(pool_p, batched_points, pool_b, batched_lengths, r,
                                                neighborhood_limits[layer])

                # # Upsample indices (with the radius of the next layer to keep wanted density)
                # up_i = batch_neighbors_kpconv(batched_points, pool_p, batched_lengths, pool_b, 2 * r,
                #                               neighborhood_limits[layer])


            else:
                # No pooling in the end of this layer, no pooling indices required
                pool_i = torch.zeros((0, 1), dtype=torch.int64)
                pool_p = torch.zeros((0, 3), dtype=torch.float32)
                pool_b = torch.zeros((0,), dtype=torch.int64)
                # up_i = torch.zeros((0, 1), dtype=torch.int64)

            # Updating input lists
            input_points += [batched_points.float()]
            input_neighbors += [conv_i.long()]
            input_pools += [pool_i.long()]
            # input_upsamples += [up_i.long()]
            input_batches_len += [batched_lengths]

            # New points for next layer
            batched_points = pool_p
            batched_lengths = pool_b

            # Update radius and reset blocks
            r_normal += config.first_subsampling_dl * config.conv_radius
            layer += 1
            layer_blocks = []

        dict_inputs['points'] = input_points
        dict_inputs['neighbors'] = input_neighbors
        dict_inputs['pools'] = input_pools
        dict_inputs['features'] = batched_features.float()
        dict_inputs['geo'] = batched_geos_list
        dict_inputs['chem'] = batched_chems_list
        dict_inputs['stack_lengths'] = input_batches_len

        if config.fusion_type == 'cascade' and config.surface2struct and config.use_superpoint:
            dict_inputs['surf2struct'] = surf2struct
        if config.fusion_type == 'msf':
            dict_inputs['surf2struct_list'] = surf2struct_list
            dict_inputs['struct2surf_list'] = struct2surf_list

    if config.dataset == 'pdbbind':
        [delattr(data, 'mess_idx') for data in batch_ligands_list
         if hasattr(data, 'mess_idx')]
        dict_inputs['ligand'] = Batch.from_data_list(batch_ligands_list)



    #
    #
    # pcd_surf1 = o3d.geometry.PointCloud()
    # pcd_surf2 = o3d.geometry.PointCloud()
    # pcd_surf3 = o3d.geometry.PointCloud()
    #
    # pcd_surf.points = o3d.utility.Vector3dVector(surf.numpy())
    #
    #
    # pcd_surf1.points = o3d.utility.Vector3dVector(dict_inputs['points'][1][:dict_inputs['stack_lengths'][1][0]].numpy() + 10)
    # pcd_surf2.points = o3d.utility.Vector3dVector(dict_inputs['points'][2][:dict_inputs['stack_lengths'][2][0]].numpy() + 20)
    # pcd_surf3.points = o3d.utility.Vector3dVector(dict_inputs['points'][3][:dict_inputs['stack_lengths'][3][0]].numpy() + 30)
    #
    #
    # pcd_surf.paint_uniform_color([1, 0, 0])
    # pcd_surf1.paint_uniform_color([0, 1, 0])
    # pcd_surf2.paint_uniform_color([0, 0, 1])
    # pcd_surf3.paint_uniform_color([0,0,0])

    return dict_inputs


