import os.path as osp
from functools import partial
from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import MLP, fps, global_max_pool, global_mean_pool, radius
from models.structure_modules import *
import esm.model.esm2
from models.surface_modules import KPFCN, KPFCN_Dense, UnaryBlock
from torch_scatter import scatter_mean
from models.ligand_modules import WLNConv
from torch_geometric.nn import global_add_pool, global_mean_pool

def radius_gaussian(sq_r, sig, eps=1e-9):
    """
    Compute a radius gaussian (gaussian of distance)
    :param sq_r: input radiuses [dn, ..., d1, d0]
    :param sig: extents of gaussians [d1, d0] or [d0] or float
    :return: gaussian of sq_r [dn, ..., d1, d0]
    """
    return torch.exp(-sq_r / (2 * sig**2 + eps))

class Linear(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 batch_norm: bool = True,
                 dropout: float = 0.0,
                 bias: bool = False,
                 leakyrelu_negative_slope: float = 0.1,
                 momentum: float = 0.2) -> nn.Module:
        super(Linear, self).__init__()

        module = []
        if batch_norm:
            module.append(nn.BatchNorm1d(in_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        module.append(nn.Dropout(dropout))
        module.append(nn.Linear(in_channels, out_channels, bias = bias))
        self.module = nn.Sequential(*module)

    def forward(self, x):
        return self.module(x)

class MLP(nn.Module):
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 batch_norm: bool,
                 dropout: float = 0.0,
                 bias: bool = True,
                 leakyrelu_negative_slope: float = 0.2,
                 momentum: float = 0.2) -> nn.Module:
        super(MLP, self).__init__()

        module = []
        if batch_norm:
            module.append(nn.BatchNorm1d(in_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        module.append(nn.Dropout(dropout))
        if mid_channels is None:
            module.append(nn.Linear(in_channels, out_channels, bias = bias))
        else:
            module.append(nn.Linear(in_channels, mid_channels, bias = bias))
        if batch_norm:
            if mid_channels is None:
                module.append(nn.BatchNorm1d(out_channels, momentum=momentum))
            else:
                module.append(nn.BatchNorm1d(mid_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        if mid_channels is None:
            module.append(nn.Dropout(dropout))
        else:
            module.append(nn.Linear(mid_channels, out_channels, bias = bias))

        self.module = nn.Sequential(*module)

    def forward(self, input):
        return self.module(input)

class BasicBlock(nn.Module):
    def __init__(self,
                 r: float,
                 l: float,
                 kernel_channels,
                 in_channels,
                 out_channels,
                 base_width: float = 16.0,
                 batch_norm: bool = True,
                 dropout: float = 0.0,
                 bias: bool = False,
                 leakyrelu_negative_slope: float = 0.1,
                 momentum = 0.2) -> nn.Module:

        super(BasicBlock, self).__init__()

        if in_channels != out_channels:
            self.identity = Linear(in_channels=in_channels,
                                  out_channels=out_channels,
                                  batch_norm=batch_norm,
                                  dropout=dropout,
                                  bias=bias,
                                  leakyrelu_negative_slope=leakyrelu_negative_slope,
                                  momentum=momentum)
        else:
            self.identity = nn.Sequential()

        width = int(out_channels * (base_width / 64.))
        self.input = MLP(in_channels=in_channels,
                         mid_channels=None,
                         out_channels=width,
                         batch_norm=batch_norm,
                         dropout=dropout,
                         bias=bias,
                         leakyrelu_negative_slope=leakyrelu_negative_slope,
                         momentum=momentum)
        self.conv = CDConv(r=r, l=l, kernel_channels=kernel_channels, in_channels=width, out_channels=width)
        self.output = Linear(in_channels=width,
                             out_channels=out_channels,
                             batch_norm=batch_norm,
                             dropout=dropout,
                             bias=bias,
                             leakyrelu_negative_slope=leakyrelu_negative_slope,
                             momentum=momentum)

    def forward(self, x, pos, seq, ori, batch):
        identity = self.identity(x)
        x = self.input(x)
        x = self.conv(x, pos, seq, ori, batch)
        out = self.output(x) + identity
        return out

class Fusion_Block(nn.Module):
    def __init__(self, channel, r, mode = 'sigma'):
        super(Fusion_Block, self).__init__()
        self.weight = nn.Linear(channel, channel)
        self.r = r
        self.norm = nn.LayerNorm(channel)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.mode = mode


    def forward(self, x, x_pad, idx, neighbor_pts, query_pts):
        x = self.weight(x)
        x = self.leaky_relu(self.norm(x))
        x_cat = torch.cat([x, x_pad], dim = 0)
        pts_cat = torch.cat([neighbor_pts, torch.zeros_like(neighbor_pts[:1, :]) + 1e6], dim = 0)
        neighbor_pts = pts_cat[idx]
        neighbor_x = x_cat[idx]

        distance = ((query_pts.unsqueeze(1) - neighbor_pts)**2).sum(-1, keepdim = True)

        if self.mode == 'linear':
            all_weights = torch.clamp(1 - distance ** 0.5 / self.r, min=1e-3)
            weighted_features = torch.mul(all_weights, neighbor_x)
        elif self.mode == 'sigma':
            sigma = self.r * 0.3
            all_weights = radius_gaussian(distance, sigma)
            all_weights = torch.clamp(all_weights, min=1e-3)
            weighted_features = torch.mul(all_weights, neighbor_x)
        # weighted_features = self.weight(weighted_features)
        weighted_features = weighted_features.sum(-2) / all_weights.sum(-2)
        # weighted_features = self.leaky_relu(weighted_features)

        return weighted_features


class ProteinF3S_Base_Func(nn.Module):
    def __init__(self,
                 cfg,
                 embedding_dim: int = 16,
                 batch_norm: bool = True,
                 dropout: float = 0.2,
                 bias: bool = False,
                 num_classes: int = 384) -> nn.Module:

        super().__init__()

        self.structure = cfg.structure
        self.surface = cfg.surface
        self.sequence = cfg.sequence

        if self.structure:
            geometric_radii = [2 * cfg.geometric_radius, 3 * cfg.geometric_radius, 4 * cfg.geometric_radius, 5 * cfg.geometric_radius]
            sequential_kernel_size = cfg.sequential_kernel_size
            kernel_channels = cfg.kernel_channels
            channels = cfg.channels
            base_width = cfg.base_width

            assert (len(geometric_radii) == len(
                channels)), "Model: 'geometric_radii' and 'channels' should have the same number of elements!"

            self.embedding = torch.nn.Embedding(num_embeddings=21, embedding_dim=embedding_dim)
            self.local_mean_pool = AvgPooling()

            layers = []
            in_channels = embedding_dim
            for i, radius in enumerate(geometric_radii):
                layers.append(BasicBlock(r=radius,
                                         l=sequential_kernel_size,
                                         kernel_channels=kernel_channels,
                                         in_channels=in_channels,
                                         out_channels=channels[i],
                                         base_width=base_width,
                                         batch_norm=batch_norm,
                                         dropout=dropout,
                                         bias=bias))
                layers.append(BasicBlock(r=radius,
                                         l=sequential_kernel_size,
                                         kernel_channels=kernel_channels,
                                         in_channels=channels[i],
                                         out_channels=channels[i],
                                         base_width=base_width,
                                         batch_norm=batch_norm,
                                         dropout=dropout,
                                         bias=bias))
                in_channels = channels[i]

            self.layers = nn.Sequential(*layers)
            self.classifier = MLP(in_channels=channels[-1],
                                  mid_channels=max(channels[-1], num_classes),
                                  out_channels=num_classes,
                                  batch_norm=batch_norm,
                                  dropout=dropout)

        if self.sequence:
            if cfg.capacity == 'tiny':
                model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
                self.repr_layers = [6]
                self.output_dim = 320
            elif cfg.capacity == 'normal':
                model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
                self.repr_layers = [12]
                self.output_dim = 480
            else:
                model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
                self.repr_layers = [33]
                self.output_dim = 1280
            self.capacity = cfg.capacity
            self.alphabet = alphabet
            self.sequence_model = model
            self.max_input_length = cfg.max_input_length

            self.classifier = MLP(in_channels=self.output_dim,
                                  mid_channels=max(self.output_dim, num_classes),
                                  out_channels=num_classes,
                                  batch_norm=batch_norm,
                                  dropout=dropout)

            self.tune = cfg.tune
            if self.tune == 'ft':
                pass
            if self.tune == 'lp':
                self.sequence_model.eval()
                for k, v in self.sequence_model.named_parameters():
                    v.requires_grad = False

        if self.surface:
            self.use_chem = cfg.use_chem
            self.use_geo = cfg.use_geo

            in_feats_dim = 1

            if self.use_chem:
                self.mini_pointnet = nn.ModuleList()
                self.mini_pointnet.append(nn.Linear(15, 21))
                self.mini_pointnet.append(nn.BatchNorm1d(21))
                self.mini_pointnet.append(nn.LeakyReLU(0.1))
                self.mini_pointnet.append(nn.Linear(21, 21))
                in_feats_dim += 21
            if self.use_geo:
                in_feats_dim += 10
            self.cat = nn.Linear(in_feats_dim, cfg.in_feats_dim)

            self.KPFCN = KPFCN(cfg)
            self.classifier = MLP(in_channels=2048,
                                  mid_channels=max(2048, num_classes),
                                  out_channels=num_classes,
                                  batch_norm=batch_norm,
                                  dropout=dropout)

        if cfg.dataset == 'pdbbind':
            self.cfg = cfg
            self.mpn = WLNConv(node_fdim=88,
                                     edge_fdim=6,
                                     depth=4,
                                     hsize=300,
                                     dropout=0.15,
                                     activation="relu",
                                     jk_pool=None)

            self.classifier = MLP(in_channels=channels[-1] + 300,
                                  mid_channels=300,
                                  out_channels=1,
                                  batch_norm=batch_norm,
                                  dropout=dropout)


    def forward(self, input):
        if self.structure:
            x = input['x']
            pos = input['pos']
            seq = input['seq']
            ori = input['ori']
            batch = input['batch']

            x, pos, seq, ori, batch = (self.embedding(x), pos, seq, ori, batch)

            for i, layer in enumerate(self.layers):
                x = layer(x, pos, seq, ori, batch)
                if i == len(self.layers) - 1:
                    x = global_mean_pool(x, batch)
                elif i % 2 == 1:
                    x, pos, seq, ori, batch = self.local_mean_pool(x, pos, seq, ori, batch)

            if self.cfg.dataset == 'pdbbind':
                lig = input['ligand']
                lig_node_emb = self.mpn(lig.x, lig.edge_index, lig.edge_attr)
                lig_emb = global_add_pool(lig_node_emb, lig.batch)
                output = torch.cat([x, lig_emb], dim=-1)
                out = self.classifier(output)
            else:
                out = self.classifier(x)

        if self.surface:
            feat = input['features'].clone().detach()
            if self.use_chem:
                chem = input['chem']
                for i, layer in enumerate(self.mini_pointnet):
                    if i == 1:
                        chem = chem.transpose(2, 1)
                        chem = layer(chem)
                        chem = chem.transpose(2, 1)
                    else:
                        chem = layer(chem)
                chem = chem.max(-2)[0]
                feat = torch.cat([chem, feat], dim=-1)
            if self.use_geo:
                geo = input['geo']
                feat = torch.cat([geo, feat], dim=-1)

            input['embedding'] = feat

            x = self.KPFCN(input)
            out = self.classifier(x)

        if self.sequence:
            batch_tokens = input['tokens']
            size = input['token_lens']

            if size.max()> self.max_input_length:
                batch_tokens = batch_tokens[:, :self.max_input_length]
                size = torch.ones_like(size) * self.max_input_length

            results = self.sequence_model(batch_tokens, repr_layers=self.repr_layers, return_contacts=False)
            token_representations = results["representations"][self.repr_layers[-1]]
            node_feature = token_representations

            global_feature = []
            index = []

            for i, i_size in enumerate(size):
                global_feature.append(token_representations[i, 1:i_size-1])
                index.append( (i * torch.ones(i_size - 2, device=node_feature.device, dtype=torch.int64)) )

            global_feature = torch.cat(global_feature)
            index = torch.cat(index)

            output = scatter_mean(global_feature, index, dim=0, dim_size=len(node_feature))
            out = self.classifier(output)

        return out


class ProteinF3S_SeqStruct_CAT_Func(nn.Module):
    def __init__(self,
                 cfg,
                 embedding_dim: int = 16,
                 batch_norm: bool = True,
                 dropout: float = 0.2,
                 bias: bool = False,
                 num_classes: int = 384) -> nn.Module:

        super().__init__()

        self.structure = cfg.structure
        self.surface = cfg.surface
        self.sequence = cfg.sequence
        assert (self.structure & self.sequence & (not self.surface))

        geometric_radii = [2 * cfg.geometric_radius, 3 * cfg.geometric_radius, 4 * cfg.geometric_radius, 5 * cfg.geometric_radius]
        sequential_kernel_size = cfg.sequential_kernel_size
        kernel_channels = cfg.kernel_channels
        channels = cfg.channels
        base_width = cfg.base_width

        assert (len(geometric_radii) == len(
            channels)), "Model: 'geometric_radii' and 'channels' should have the same number of elements!"

        self.embedding = torch.nn.Embedding(num_embeddings=21, embedding_dim=embedding_dim)
        self.local_mean_pool = AvgPooling()

        layers = []
        in_channels = embedding_dim
        for i, radius in enumerate(geometric_radii):
            layers.append(BasicBlock(r=radius,
                                     l=sequential_kernel_size,
                                     kernel_channels=kernel_channels,
                                     in_channels=in_channels,
                                     out_channels=channels[i],
                                     base_width=base_width,
                                     batch_norm=batch_norm,
                                     dropout=dropout,
                                     bias=bias))
            layers.append(BasicBlock(r=radius,
                                     l=sequential_kernel_size,
                                     kernel_channels=kernel_channels,
                                     in_channels=channels[i],
                                     out_channels=channels[i],
                                     base_width=base_width,
                                     batch_norm=batch_norm,
                                     dropout=dropout,
                                     bias=bias))
            in_channels = channels[i]

        self.layers = nn.Sequential(*layers)


        if cfg.capacity == 'tiny':
            model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            self.repr_layers = [6]
            self.output_dim = 320
        elif cfg.capacity == 'normal':
            model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            self.repr_layers = [12]
            self.output_dim = 480
        else:
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.repr_layers = [33]
            self.output_dim = 1280
        self.capacity = cfg.capacity
        self.alphabet = alphabet
        self.sequence_model = model
        self.max_input_length = cfg.max_input_length

        self.tune = cfg.tune
        if self.tune == 'ft':
            pass
        if self.tune == 'lp':
            self.sequence_model.eval()
            for k, v in self.sequence_model.named_parameters():
                v.requires_grad = False

        self.classifier = MLP(in_channels=channels[-1] + self.output_dim,
                      mid_channels=max(channels[-1] + self.output_dim, num_classes),
                      out_channels=num_classes,
                      batch_norm=batch_norm,
                      dropout=dropout)


    def forward(self, input):

        x = input['x']
        pos = input['pos']
        seq = input['seq']
        ori = input['ori']
        batch = input['batch']

        x, pos, seq, ori, batch = (self.embedding(x), pos, seq, ori, batch)

        for i, layer in enumerate(self.layers):
            x = layer(x, pos, seq, ori, batch)
            if i == len(self.layers) - 1:
                x = global_mean_pool(x, batch)
            elif i % 2 == 1:
                x, pos, seq, ori, batch = self.local_mean_pool(x, pos, seq, ori, batch)
        output_structure = x

        batch_tokens = input['tokens']
        size = input['token_lens']

        if size.max()> self.max_input_length:
            batch_tokens = batch_tokens[:, :self.max_input_length]
            size = torch.ones_like(size) * self.max_input_length

        results = self.sequence_model(batch_tokens, repr_layers=self.repr_layers, return_contacts=False)
        token_representations = results["representations"][self.repr_layers[-1]]
        node_feature = token_representations

        global_feature = []
        index = []

        for i, i_size in enumerate(size):
            global_feature.append(token_representations[i, 1:i_size-1])
            index.append( (i * torch.ones(i_size - 2, device=node_feature.device, dtype=torch.int64)) )

        global_feature = torch.cat(global_feature)
        index = torch.cat(index)

        output_sequence = scatter_mean(global_feature, index, dim=0, dim_size=len(node_feature))

        output = torch.cat([output_sequence, output_structure], dim = -1)
        out = self.classifier(output)

        return out


# class ProteinF3S_SeqStruct_MSF_Func(nn.Module):
#     def __init__(self,
#                  cfg,
#                  embedding_dim: int = 16,
#                  batch_norm: bool = True,
#                  dropout: float = 0.2,
#                  bias: bool = False,
#                  num_classes: int = 384) -> nn.Module:
#
#         super().__init__()
#
#         self.structure = cfg.structure
#         self.surface = cfg.surface
#         self.sequence = cfg.sequence
#         assert (self.structure & self.sequence & (not self.surface))
#
#         if cfg.capacity == 'tiny':
#             model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
#             self.repr_layers = [3,4,5,6]
#             self.output_dim = 320
#         elif cfg.capacity == 'normal':
#             model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
#             self.repr_layers = [3, 6, 9, 12]
#             self.output_dim = 480
#         else:
#             model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
#             self.repr_layers = [9, 17, 25, 33]
#             self.output_dim = 1280
#         self.capacity = cfg.capacity
#         self.alphabet = alphabet
#         self.sequence_model = model
#         self.max_input_length = cfg.max_input_length
#
#         self.tune = cfg.tune
#         if self.tune == 'lp':
#             self.sequence_model.eval()
#             for k, v in self.sequence_model.named_parameters():
#                 v.requires_grad = False
#         self.padding_embedding = [nn.Parameter(torch.ones([1, self.output_dim]).float(), requires_grad=True),
#                                   nn.Parameter(torch.ones([1, self.output_dim]).float(), requires_grad=True),
#                                   nn.Parameter(torch.ones([1, self.output_dim]).float(), requires_grad=True)]
#
#         geometric_radii = [2 * cfg.geometric_radius, 3 * cfg.geometric_radius, 4 * cfg.geometric_radius, 5 * cfg.geometric_radius]
#         sequential_kernel_size = cfg.sequential_kernel_size
#         kernel_channels = cfg.kernel_channels
#         channels = cfg.channels
#         base_width = cfg.base_width
#
#         assert (len(geometric_radii) == len(
#             channels)), "Model: 'geometric_radii' and 'channels' should have the same number of elements!"
#
#         self.embedding = torch.nn.Embedding(num_embeddings=21, embedding_dim=embedding_dim)
#         self.local_mean_pool = AvgPooling()
#
#         layers = []
#         in_channels = embedding_dim
#         for i, radius in enumerate(geometric_radii):
#             layers.append(BasicBlock(r=radius,
#                                      l=sequential_kernel_size,
#                                      kernel_channels=kernel_channels,
#                                      in_channels=in_channels,
#                                      out_channels=channels[i],
#                                      base_width=base_width,
#                                      batch_norm=batch_norm,
#                                      dropout=dropout,
#                                      bias=bias))
#             layers.append(BasicBlock(r=radius,
#                                      l=sequential_kernel_size,
#                                      kernel_channels=kernel_channels,
#                                      in_channels=channels[i],
#                                      out_channels=channels[i],
#                                      base_width=base_width,
#                                      batch_norm=batch_norm,
#                                      dropout=dropout,
#                                      bias=bias))
#             in_channels = channels[i]
#
#         self.layers = nn.Sequential(*layers)
#
#
#         self.classifier = MLP(in_channels=channels[-1] + self.output_dim,
#                       mid_channels=max(channels[-1] + self.output_dim, num_classes),
#                       out_channels=num_classes,
#                       batch_norm=batch_norm,
#                       dropout=dropout)
#
#
#     def forward(self, input):
#
#         batch_tokens = input['tokens']
#         size = input['token_lens']
#
#         if size.max() > self.max_input_length:
#             batch_tokens = batch_tokens[:, :self.max_input_length]
#             size = torch.ones_like(size) * self.max_input_length
#
#         results = self.sequence_model(batch_tokens, repr_layers=self.repr_layers, return_contacts=False)
#         token_representations = results["representations"][self.repr_layers[-1]]
#         node_feature = token_representations
#
#         global_feature = []
#         index = []
#
#         for i, i_size in enumerate(size):
#             global_feature.append(token_representations[i, 1:i_size - 1])
#             index.append((i * torch.ones(i_size - 2, device=node_feature.device, dtype=torch.int64)))
#
#         global_feature = torch.cat(global_feature)
#         index = torch.cat(index)
#
#         output_sequence = scatter_mean(global_feature, index, dim=0, dim_size=len(node_feature))
#
#
#         x = input['x']
#         pos = input['pos']
#         seq = input['seq']
#         ori = input['ori']
#         batch = input['batch']
#
#         sequence_representation = self.resize_sequence(results["representations"], seq, input['token_lens'])
#
#         x, pos, seq, ori, batch = (self.embedding(x), pos, seq, ori, batch)
#
#         for i, layer in enumerate(self.layers):
#             x = layer(x, pos, seq, ori, batch)
#             if i == len(self.layers) - 1:
#                 x = global_mean_pool(x, batch)
#             elif i % 2 == 1:
#                 # sequence_representation = results["representations"][self.repr_layers[i//2]]
#                 # sequence_representation = self.resize_sequence(sequence_representation, input['token_lens'])
#                 # x = torch.cat([x, sequence_representation], dim=-1)
#
#                 x, pos, seq, ori, batch = self.local_mean_pool(x, pos, seq, ori, batch)
#         output_structure = x
#
#
#
#         output = torch.cat([output_sequence, output_structure], dim = -1)
#         out = self.classifier(output)
#
#         return out
#
#     def resize_sequence(self, sequence_representation, seq, len):
#         sequence_representation_resized = []
#         if (len > self.max_input_length).any():
#             for pad_i, i in self.repr_layers[:-1]:
#                 sequence_representation_resized_i = []
#                 for j in range(sequence_representation[i].shape[0]):
#                     if len[j] <= self.max_input_length - 2:
#                         sequence_representation_resized_i.append(sequence_representation[i][j][1:1+len[j]])
#                     else:
#                         sequence_representation_resized_i.append(torch.cat([sequence_representation[i][j][1:-1],
#                                                                            self.padding_embedding[pad_i].repeat(len[j]-1022, 1) ], dim=0))
#                 sequence_representation_resized.append(torch.cat(sequence_representation_resized_i, dim = 0))
#         else:
#             for i in self.repr_layers[:-1]:
#                 sequence_representation_resized_i = []
#                 for j in range(sequence_representation[i].shape[0]):
#                     sequence_representation_resized_i.append(sequence_representation[i][j][1:1+len[j]])
#                 sequence_representation_resized.append(torch.cat(sequence_representation_resized_i, dim = 0))
#
#         for i in range(len(sequence_representation)):
#             idx = torch.div(seq.squeeze(1), 2*i, rounding_mode='floor')
#             idx = torch.cat([idx, idx[-1].view((1,))])
#
#             idx = (idx[0:-1] != idx[1:]).to(torch.float32)
#             idx = torch.cumsum(idx, dim=0) - idx
#             idx = idx.to(torch.int64)
#
#             sequence_representation
#             x = scatter_mean(src=x, index=idx, dim=0)


class ProteinF3S_SeqSurf_CAT_Func(nn.Module):
    def __init__(self,
                 cfg,
                 embedding_dim: int = 16,
                 batch_norm: bool = True,
                 dropout: float = 0.2,
                 bias: bool = False,
                 num_classes: int = 384) -> nn.Module:

        super().__init__()

        self.structure = cfg.structure
        self.surface = cfg.surface
        self.sequence = cfg.sequence
        assert (self.structure & self.sequence & self.surface)

        if self.sequence:
            if cfg.capacity == 'tiny':
                model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
                self.repr_layers = [6]
                self.output_dim = 320
            elif cfg.capacity == 'normal':
                model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
                self.repr_layers = [12]
                self.output_dim = 480
            else:
                model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
                self.repr_layers = [33]
                self.output_dim = 1280
            self.capacity = cfg.capacity
            self.alphabet = alphabet
            self.sequence_model = model
            self.max_input_length = cfg.max_input_length

            self.tune = cfg.tune
            if self.tune == 'ft':
                pass
            if self.tune == 'lp':
                self.sequence_model.eval()
                for k, v in self.sequence_model.named_parameters():
                    v.requires_grad = False

        if self.surface:
            self.use_chem = cfg.use_chem
            self.use_geo = cfg.use_geo

            in_feats_dim = 1

            if self.use_chem:
                self.mini_pointnet = nn.ModuleList()
                self.mini_pointnet.append(nn.Linear(15, 21))
                self.mini_pointnet.append(nn.BatchNorm1d(21))
                self.mini_pointnet.append(nn.LeakyReLU(0.1))
                self.mini_pointnet.append(nn.Linear(21, 21))
                in_feats_dim += 21
            if self.use_geo:
                in_feats_dim += 10
            self.cat = nn.Linear(in_feats_dim, cfg.in_feats_dim)

            self.KPFCN = KPFCN(cfg)

        self.classifier = MLP(in_channels=2048 + self.output_dim,
                                  mid_channels=max(2048 + self.output_dim, num_classes),
                                  out_channels=num_classes,
                                  batch_norm=batch_norm,
                                  dropout=dropout)



    def forward(self, input):
        batch_tokens = input['tokens']
        size = input['token_lens']

        if size.max() > self.max_input_length:
            batch_tokens = batch_tokens[:, :self.max_input_length]
            size = torch.ones_like(size) * self.max_input_length

        results = self.sequence_model(batch_tokens, repr_layers=self.repr_layers, return_contacts=False)
        token_representations = results["representations"][self.repr_layers[-1]]
        node_feature = token_representations

        global_feature = []
        index = []

        for i, i_size in enumerate(size):
            global_feature.append(token_representations[i, 1:i_size - 1])
            index.append((i * torch.ones(i_size - 2, device=node_feature.device, dtype=torch.int64)))

        global_feature = torch.cat(global_feature)
        index = torch.cat(index)

        output_seq = scatter_mean(global_feature, index, dim=0, dim_size=len(node_feature))

        feat = input['features'].clone().detach()
        if self.use_chem:
            chem = input['chem']
            for i, layer in enumerate(self.mini_pointnet):
                if i == 1:
                    chem = chem.transpose(2, 1)
                    chem = layer(chem)
                    chem = chem.transpose(2, 1)
                else:
                    chem = layer(chem)
            chem = chem.max(-2)[0]
            feat = torch.cat([chem, feat], dim=-1)
        if self.use_geo:
            geo = input['geo']
            feat = torch.cat([geo, feat], dim=-1)

        input['embedding'] = feat

        out_surf = self.KPFCN(input)

        output = torch.cat([out_surf, output_seq], dim=-1)
        out = self.classifier(output)

        return out


class ProteinF3S_SurfStruct_CAT_Func(nn.Module):
    def __init__(self,
                 cfg,
                 embedding_dim: int = 16,
                 batch_norm: bool = True,
                 dropout: float = 0.2,
                 bias: bool = False,
                 num_classes: int = 384) -> nn.Module:

        super().__init__()

        self.structure = cfg.structure
        self.surface = cfg.surface
        self.sequence = cfg.sequence
        assert (self.structure & self.surface & (not self.sequence))

        geometric_radii = [2 * cfg.geometric_radius, 3 * cfg.geometric_radius, 4 * cfg.geometric_radius, 5 * cfg.geometric_radius]
        sequential_kernel_size = cfg.sequential_kernel_size
        kernel_channels = cfg.kernel_channels
        channels = cfg.channels
        base_width = cfg.base_width

        assert (len(geometric_radii) == len(
            channels)), "Model: 'geometric_radii' and 'channels' should have the same number of elements!"

        self.embedding = torch.nn.Embedding(num_embeddings=21, embedding_dim=embedding_dim)
        self.local_mean_pool = AvgPooling()

        layers = []
        in_channels = embedding_dim
        for i, radius in enumerate(geometric_radii):
            layers.append(BasicBlock(r=radius,
                                     l=sequential_kernel_size,
                                     kernel_channels=kernel_channels,
                                     in_channels=in_channels,
                                     out_channels=channels[i],
                                     base_width=base_width,
                                     batch_norm=batch_norm,
                                     dropout=dropout,
                                     bias=bias))
            layers.append(BasicBlock(r=radius,
                                     l=sequential_kernel_size,
                                     kernel_channels=kernel_channels,
                                     in_channels=channels[i],
                                     out_channels=channels[i],
                                     base_width=base_width,
                                     batch_norm=batch_norm,
                                     dropout=dropout,
                                     bias=bias))
            in_channels = channels[i]

        self.layers = nn.Sequential(*layers)

        self.use_chem = cfg.use_chem
        self.use_geo = cfg.use_geo

        in_feats_dim = 1

        if self.use_chem:
            self.mini_pointnet = nn.ModuleList()
            self.mini_pointnet.append(nn.Linear(15, 21))
            self.mini_pointnet.append(nn.BatchNorm1d(21))
            self.mini_pointnet.append(nn.LeakyReLU(0.1))
            self.mini_pointnet.append(nn.Linear(21, 21))
            in_feats_dim += 21
        if self.use_geo:
            in_feats_dim += 10
        self.cat = nn.Linear(in_feats_dim, cfg.in_feats_dim)

        self.KPFCN = KPFCN(cfg)
        self.classifier = MLP(in_channels=2048 + channels[-1],
                              mid_channels=max(2048 + channels[-1], num_classes),
                              out_channels=num_classes,
                              batch_norm=batch_norm,
                              dropout=dropout)


    def forward(self, input):

        x = input['x']
        pos = input['pos']
        seq = input['seq']
        ori = input['ori']
        batch = input['batch']

        x, pos, seq, ori, batch = (self.embedding(x), pos, seq, ori, batch)

        for i, layer in enumerate(self.layers):
            x = layer(x, pos, seq, ori, batch)
            if i == len(self.layers) - 1:
                x = global_mean_pool(x, batch)
            elif i % 2 == 1:
                x, pos, seq, ori, batch = self.local_mean_pool(x, pos, seq, ori, batch)
        out_struct = x

        feat = input['features'].clone().detach()
        if self.use_chem:
            chem = input['chem']
            for i, layer in enumerate(self.mini_pointnet):
                if i == 1:
                    chem = chem.transpose(2, 1)
                    chem = layer(chem)
                    chem = chem.transpose(2, 1)
                else:
                    chem = layer(chem)
            chem = chem.max(-2)[0]
            feat = torch.cat([chem, feat], dim=-1)
        if self.use_geo:
            geo = input['geo']
            feat = torch.cat([geo, feat], dim=-1)

        input['embedding'] = feat

        out_surf = self.KPFCN(input)

        output = torch.cat([out_surf, out_struct], dim = -1)
        out = self.classifier(output)

        return out


class ProteinF3S_Surf2Struct_CAS_Func(nn.Module):
    def __init__(self,
                 cfg,
                 embedding_dim: int = 16,
                 batch_norm: bool = True,
                 dropout: float = 0.2,
                 bias: bool = False,
                 num_classes: int = 384) -> nn.Module:

        super().__init__()

        self.structure = cfg.structure
        self.surface = cfg.surface
        self.sequence = cfg.sequence
        assert (self.structure & self.surface & (not self.sequence))

        geometric_radii = [2 * cfg.geometric_radius, 3 * cfg.geometric_radius, 4 * cfg.geometric_radius, 5 * cfg.geometric_radius]
        sequential_kernel_size = cfg.sequential_kernel_size
        kernel_channels = cfg.kernel_channels
        channels = cfg.channels
        base_width = cfg.base_width

        assert (len(geometric_radii) == len(
            channels)), "Model: 'geometric_radii' and 'channels' should have the same number of elements!"

        self.embedding = torch.nn.Embedding(num_embeddings=21, embedding_dim=embedding_dim)
        self.local_mean_pool = AvgPooling()

        layers = []
        in_channels = embedding_dim
        for i, radius in enumerate(geometric_radii):
            layers.append(BasicBlock(r=radius,
                                     l=sequential_kernel_size,
                                     kernel_channels=kernel_channels,
                                     in_channels=in_channels,
                                     out_channels=channels[i],
                                     base_width=base_width,
                                     batch_norm=batch_norm,
                                     dropout=dropout,
                                     bias=bias))
            layers.append(BasicBlock(r=radius,
                                     l=sequential_kernel_size,
                                     kernel_channels=kernel_channels,
                                     in_channels=channels[i],
                                     out_channels=channels[i],
                                     base_width=base_width,
                                     batch_norm=batch_norm,
                                     dropout=dropout,
                                     bias=bias))
            in_channels = channels[i]

        self.layers = nn.Sequential(*layers)

        self.use_chem = cfg.use_chem
        self.use_geo = cfg.use_geo

        in_feats_dim = 1

        if self.use_chem:
            self.mini_pointnet = nn.ModuleList()
            self.mini_pointnet.append(nn.Linear(15, 21))
            self.mini_pointnet.append(nn.BatchNorm1d(21))
            self.mini_pointnet.append(nn.LeakyReLU(0.1))
            self.mini_pointnet.append(nn.Linear(21, 21))
            in_feats_dim += 21
        if self.use_geo:
            in_feats_dim += 10
        self.cat = nn.Linear(in_feats_dim, cfg.in_feats_dim)

        if cfg.use_superpoint:
            cfg.architectures.pop()
            self.KPFCN = KPFCN(cfg)
        else:
            cfg.architectures = ['simple',
                 'resnetb',
                 'resnetb_strided',
                 'resnetb',
                 'resnetb',
                 'resnetb_strided',
                 'resnetb',
                 'resnetb',
                 'resnetb_strided',
                 'resnetb',
                 'resnetb',
                'nearest_upsample',
                'unary',
                 'nearest_upsample',
                 'unary',
                'nearest_upsample',
                'last_unary'
            ]
            self.KPFCN = KPFCN_Dense(cfg)
        self.fusion = nn.Linear(32, 128)

        self.classifier = MLP(in_channels=channels[-1],
                              mid_channels=max(channels[-1], num_classes),
                              out_channels=num_classes,
                              batch_norm=batch_norm,
                              dropout=dropout)


    def forward(self, input):

        feat = input['features'].clone().detach()
        if self.use_chem:
            chem = input['chem']
            for i, layer in enumerate(self.mini_pointnet):
                if i == 1:
                    chem = chem.transpose(2, 1)
                    chem = layer(chem)
                    chem = chem.transpose(2, 1)
                else:
                    chem = layer(chem)
            chem = chem.max(-2)[0]
            feat = torch.cat([chem, feat], dim=-1)
        if self.use_geo:
            geo = input['geo']
            feat = torch.cat([geo, feat], dim=-1)

        input['embedding'] = feat

        out_surf = self.KPFCN(input)

        x = input['x']
        pos = input['pos']
        seq = input['seq']
        ori = input['ori']
        batch = input['batch']

        # embedding = torch.cat([self.embedding(x), out_surf], dim = -1)

        x, pos, seq, ori, batch = (self.embedding(x), pos, seq, ori, batch)

        for i, layer in enumerate(self.layers):
            if i == 0:
                identity = layer.identity(x)
                x = layer.input(x)
                x = self.fusion(out_surf[input['surf2struct']]) + x
                x = layer.conv(x, pos, seq, ori, batch)
                x = layer.output(x) + identity
            else:
                x = layer(x, pos, seq, ori, batch)
            if i == len(self.layers) - 1:
                x = global_mean_pool(x, batch)
            elif i % 2 == 1:
                x, pos, seq, ori, batch = self.local_mean_pool(x, pos, seq, ori, batch)
        out_struct = x

        out = self.classifier(out_struct)

        return out

# previous
# class ProteinF3S_Surf2Struct_MSF_Func(nn.Module):
#     def __init__(self,
#                  cfg,
#                  embedding_dim: int = 16,
#                  batch_norm: bool = True,
#                  dropout: float = 0.2,
#                  bias: bool = False,
#                  num_classes: int = 384) -> nn.Module:
#
#         super().__init__()
#
#         self.structure = cfg.structure
#         self.surface = cfg.surface
#         self.sequence = cfg.sequence
#         assert (self.structure & self.surface & (not self.sequence))
#
#         geometric_radii = [2 * cfg.geometric_radius, 3 * cfg.geometric_radius, 4 * cfg.geometric_radius, 5 * cfg.geometric_radius]
#         sequential_kernel_size = cfg.sequential_kernel_size
#         kernel_channels = cfg.kernel_channels
#         channels = cfg.channels
#         base_width = cfg.base_width
#
#         assert (len(geometric_radii) == len(
#             channels)), "Model: 'geometric_radii' and 'channels' should have the same number of elements!"
#
#         # fusion
#         self.surf2struct_blocks = nn.ModuleList()
#         self.struct2surf_blocks = nn.ModuleList()
#
#         # struct
#         self.embedding = torch.nn.Embedding(num_embeddings=21, embedding_dim=embedding_dim)
#         self.local_mean_pool = AvgPooling()
#
#         layers = []
#         in_channels = embedding_dim
#         for i, radius in enumerate(geometric_radii):
#             layers.append(BasicBlock(r=radius,
#                                      l=sequential_kernel_size,
#                                      kernel_channels=kernel_channels,
#                                      in_channels=in_channels,
#                                      out_channels=channels[i],
#                                      base_width=base_width,
#                                      batch_norm=batch_norm,
#                                      dropout=dropout,
#                                      bias=bias))
#             layers.append(BasicBlock(r=radius,
#                                      l=sequential_kernel_size,
#                                      kernel_channels=kernel_channels,
#                                      in_channels=channels[i],
#                                      out_channels=channels[i],
#                                      base_width=base_width,
#                                      batch_norm=batch_norm,
#                                      dropout=dropout,
#                                      bias=bias))
#             in_channels = channels[i]
#
#
#             self.surf2struct_blocks.append(
#                 nn.Sequential(
#                     nn.Linear(in_channels, in_channels),
#                     nn.LayerNorm(in_channels),
#                     nn.LeakyReLU(0.1),
#             ))
#             self.surf2struct_blocks.append(
#                 nn.Sequential(
#                     nn.Linear(in_channels * 2, in_channels, 1),
#                     nn.BatchNorm1d(in_channels),
#                     nn.LeakyReLU(0.1),
#                     nn.Linear(in_channels, in_channels, 1)
#                 )
#             )
#             self.surf2struct_blocks.append(nn.Linear(in_channels, in_channels))
#
#
#             self.struct2surf_blocks.append(
#                 nn.Sequential(
#                     nn.Linear(in_channels, in_channels),
#                     nn.LayerNorm(in_channels),
#                     nn.LeakyReLU(0.1),
#                 )
#             )
#             self.struct2surf_blocks.append(
#                 nn.Sequential(
#                     nn.Linear(in_channels * 2, in_channels, 1),
#                     nn.BatchNorm1d(in_channels),
#                     nn.LeakyReLU(0.1),
#                     nn.Linear(in_channels, in_channels, 1)
#                 )
#             )
#             self.struct2surf_blocks.append(nn.Linear(in_channels, in_channels))
#
#
#
#         self.layers = nn.Sequential(*layers)
#
#         # surface
#         self.use_chem = cfg.use_chem
#         self.use_geo = cfg.use_geo
#
#         in_feats_dim = 1
#
#         if self.use_chem:
#             self.mini_pointnet = nn.ModuleList()
#             self.mini_pointnet.append(nn.Linear(15, 21))
#             self.mini_pointnet.append(nn.BatchNorm1d(21))
#             self.mini_pointnet.append(nn.LeakyReLU(0.1))
#             self.mini_pointnet.append(nn.Linear(21, 21))
#             in_feats_dim += 21
#         if self.use_geo:
#             in_feats_dim += 10
#
#         self.KPFCN = KPFCN(cfg)
#
#         self.classifier = MLP(in_channels=channels[-1],
#                               mid_channels=max(channels[-1], num_classes),
#                               out_channels=num_classes,
#                               batch_norm=batch_norm,
#                               dropout=dropout)
#
#
#     def forward(self, input):
#
#         feat = input['features'].clone().detach()
#         if self.use_chem:
#             chem = input['chem']
#             for i, layer in enumerate(self.mini_pointnet):
#                 if i == 1:
#                     chem = chem.transpose(2, 1)
#                     chem = layer(chem)
#                     chem = chem.transpose(2, 1)
#                 else:
#                     chem = layer(chem)
#             chem = chem.max(-2)[0]
#             feat = torch.cat([chem, feat], dim=-1)
#         if self.use_geo:
#             geo = input['geo']
#             feat = torch.cat([geo, feat], dim=-1)
#         x_surf = feat
#
#         x = input['x']
#         pos = input['pos']
#         seq = input['seq']
#         ori = input['ori']
#         batch = input['batch']
#         x_struct, pos, seq, ori, batch = (self.embedding(x), pos, seq, ori, batch)
#
#
#         for i, layer in enumerate(self.layers):
#             x_struct = layer(x_struct, pos, seq, ori, batch)
#             if i == len(self.layers) - 1:
#                 x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 0: 3 * (i // 2) + 1][0](x_surf, input)
#                 x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 1: 3 * (i // 2) + 2][0](x_surf, input)
#
#                 x_surf2struct = self.surf2struct_blocks[3 * (i // 2) + 0: 3 * (i // 2) + 1][0](
#                     x_surf[input['surf2struct_list'][i // 2]]).mean(-2)
#                 x_surf2struct = torch.cat([x_struct, x_surf2struct], dim=-1)
#                 x_surf2struct = self.surf2struct_blocks[3 * (i // 2) + 1: 3 * (i // 2) + 2][0](x_surf2struct)
#                 x_struct = self.surf2struct_blocks[3 * (i // 2) + 2: 3 * (i // 2) + 3][0](x_struct) + x_surf2struct
#
#                 x_struct = global_mean_pool(x_struct, batch)
#             elif i % 2 == 1:
#
#                 x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 0: 3 * (i // 2) + 1][0](x_surf, input)
#                 x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 1: 3 * (i // 2) + 2][0](x_surf, input)
#
#                 # surface to struct fusion
#                 x_surf2struct = self.surf2struct_blocks[3 * (i // 2) + 0: 3 * (i // 2) + 1][0](x_surf[input['surf2struct_list'][i // 2]]).mean(-2)
#                 x_surf2struct = torch.cat([x_struct, x_surf2struct], dim = -1)
#                 x_surf2struct = self.surf2struct_blocks[3 * (i // 2) + 1: 3 * (i // 2) + 2][0](x_surf2struct)
#
#                 # struct to surface fusion
#                 x_struct2surf = self.struct2surf_blocks[3 * (i // 2) + 0: 3 * (i // 2) + 1][0](x_struct[input['struct2surf_list'][i // 2]]).mean(-2)
#                 x_struct2surf = torch.cat([x_surf, x_struct2surf], dim=-1)
#                 x_struct2surf = self.struct2surf_blocks[3 * (i // 2) + 1: 3 * (i // 2) + 2][0](x_struct2surf)
#
#
#                 # update fusion results
#                 x_struct = self.surf2struct_blocks[3 * (i // 2) + 2: 3 * (i // 2) + 3][0](x_struct) + x_surf2struct
#                 x_surf = self.struct2surf_blocks[3 * (i // 2) + 2: 3 * (i // 2) + 3][0](x_surf) + x_struct2surf
#
#
#                 # downsample
#                 x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 2: 3 * (i // 2) + 3][0](x_surf, input)
#                 x_struct, pos, seq, ori, batch = self.local_mean_pool(x_struct, pos, seq, ori, batch)
#
#
#         output = x_struct
#         out = self.classifier(output)
#
#         return out



# sparse
# class ProteinF3S_Surf2Struct_MSF_Func(nn.Module):
#     def __init__(self,
#                  cfg,
#                  embedding_dim: int = 16,
#                  batch_norm: bool = True,
#                  dropout: float = 0.2,
#                  bias: bool = False,
#                  num_classes: int = 384) -> nn.Module:
#
#         super().__init__()
#
#         self.structure = cfg.structure
#         self.surface = cfg.surface
#         self.sequence = cfg.sequence
#         assert (self.structure & self.surface & (not self.sequence))
#
#         geometric_radii = [2 * cfg.geometric_radius, 3 * cfg.geometric_radius, 4 * cfg.geometric_radius, 5 * cfg.geometric_radius]
#         sequential_kernel_size = cfg.sequential_kernel_size
#         kernel_channels = cfg.kernel_channels
#         channels = cfg.channels
#         base_width = cfg.base_width
#
#         assert (len(geometric_radii) == len(
#             channels)), "Model: 'geometric_radii' and 'channels' should have the same number of elements!"
#
#         # fusion
#         self.surf2struct_blocks = nn.ModuleList()
#         self.struct2surf_blocks = nn.ModuleList()
#
#         # struct
#         self.embedding = torch.nn.Embedding(num_embeddings=21, embedding_dim=embedding_dim)
#         self.local_mean_pool = AvgPooling()
#
#         layers = []
#         self.padding_surf2struct = []
#         self.padding_struct2surf = []
#         in_channels = embedding_dim
#         for i, radius in enumerate(geometric_radii):
#             layers.append(BasicBlock(r=radius,
#                                      l=sequential_kernel_size,
#                                      kernel_channels=kernel_channels,
#                                      in_channels=in_channels,
#                                      out_channels=channels[i],
#                                      base_width=base_width,
#                                      batch_norm=batch_norm,
#                                      dropout=dropout,
#                                      bias=bias))
#             layers.append(BasicBlock(r=radius,
#                                      l=sequential_kernel_size,
#                                      kernel_channels=kernel_channels,
#                                      in_channels=channels[i],
#                                      out_channels=channels[i],
#                                      base_width=base_width,
#                                      batch_norm=batch_norm,
#                                      dropout=dropout,
#                                      bias=bias))
#             in_channels = channels[i]
#
#             self.surf2struct_blocks.append(
#                 nn.Sequential(
#                     nn.Linear(in_channels, in_channels),
#                     nn.LayerNorm(in_channels),
#                     nn.LeakyReLU(0.1),
#             ))
#             self.surf2struct_blocks.append(
#                 nn.Sequential(
#                     nn.Linear(in_channels * 2, in_channels, 1),
#                     nn.BatchNorm1d(in_channels),
#                     nn.LeakyReLU(0.1),
#                     nn.Linear(in_channels, in_channels, 1)
#                 )
#             )
#             self.surf2struct_blocks.append(nn.Linear(in_channels, in_channels))
#
#
#             self.struct2surf_blocks.append(
#                 nn.Sequential(
#                     nn.Linear(in_channels, in_channels),
#                     nn.LayerNorm(in_channels),
#                     nn.LeakyReLU(0.1),
#                 )
#             )
#             self.struct2surf_blocks.append(
#                 nn.Sequential(
#                     nn.Linear(in_channels * 2, in_channels, 1),
#                     nn.BatchNorm1d(in_channels),
#                     nn.LeakyReLU(0.1),
#                     nn.Linear(in_channels, in_channels, 1)
#                 )
#             )
#             self.struct2surf_blocks.append(nn.Linear(in_channels, in_channels))
#
#             self.padding_surf2struct.append(
#                 nn.init.xavier_uniform_(nn.Parameter(torch.zeros(1, in_channels))).cuda()
#             )
#             self.padding_struct2surf.append(
#                 nn.init.xavier_uniform_(nn.Parameter(torch.zeros(1, in_channels))).cuda()
#             )
#
#
#         self.layers = nn.Sequential(*layers)
#
#         # surface
#         self.use_chem = cfg.use_chem
#         self.use_geo = cfg.use_geo
#
#         in_feats_dim = 1
#
#         if self.use_chem:
#             self.mini_pointnet = nn.ModuleList()
#             self.mini_pointnet.append(nn.Linear(15, 21))
#             self.mini_pointnet.append(nn.BatchNorm1d(21))
#             self.mini_pointnet.append(nn.LeakyReLU(0.1))
#             self.mini_pointnet.append(nn.Linear(21, 21))
#             in_feats_dim += 21
#         if self.use_geo:
#             in_feats_dim += 10
#
#         self.KPFCN = KPFCN(cfg)
#
#         self.classifier = MLP(in_channels=1 * channels[-1],
#                               mid_channels=max(channels[-1], num_classes),
#                               out_channels=num_classes,
#                               batch_norm=batch_norm,
#                               dropout=dropout)
#
#
#     def forward(self, input):
#
#         feat = input['features'].clone().detach()
#         if self.use_chem:
#             chem = input['chem']
#             for i, layer in enumerate(self.mini_pointnet):
#                 if i == 1:
#                     chem = chem.transpose(2, 1)
#                     chem = layer(chem)
#                     chem = chem.transpose(2, 1)
#                 else:
#                     chem = layer(chem)
#             chem = chem.max(-2)[0]
#             feat = torch.cat([chem, feat], dim=-1)
#         if self.use_geo:
#             geo = input['geo']
#             feat = torch.cat([geo, feat], dim=-1)
#         x_surf = feat
#
#         x = input['x']
#         pos = input['pos']
#         seq = input['seq']
#         ori = input['ori']
#         batch = input['batch']
#         x_struct, pos, seq, ori, batch = (self.embedding(x), pos, seq, ori, batch)
#
#
#         for i, layer in enumerate(self.layers):
#             x_struct = layer(x_struct, pos, seq, ori, batch)
#             if i == len(self.layers) - 1:
#                 # x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 0: 3 * (i // 2) + 1][0](x_surf, input)
#                 # x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 1: 3 * (i // 2) + 2][0](x_surf, input)
#                 # x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 2: 3 * (i // 2) + 3][0](x_surf, input)
#
#                 x_struct = global_mean_pool(x_struct, batch)
#
#                 # x_struct = torch.cat([x_struct, x_surf], dim=-1)
#             elif i % 2 == 1:
#
#                 x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 0: 3 * (i // 2) + 1][0](x_surf, input)
#                 x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 1: 3 * (i // 2) + 2][0](x_surf, input)
#
#                 # downsample
#                 x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 2: 3 * (i // 2) + 3][0](x_surf, input)
#                 x_struct, pos, seq, ori, batch = self.local_mean_pool(x_struct, pos, seq, ori, batch)
#
#                 # surface to struct fusion
#                 x_surf2struct = torch.cat([x_surf, self.padding_surf2struct[i // 2]], dim = 0)
#                 x_struct2surf = x_surf2struct[input['surf2struct_list'][i // 2]]
#                 x_surf2struct = self.surf2struct_blocks[3 * (i // 2) + 0: 3 * (i // 2) + 1][0](x_struct2surf).max(-2)[0]
#                 x_surf2struct = torch.cat([x_struct, x_surf2struct], dim = -1)
#                 x_surf2struct = self.surf2struct_blocks[3 * (i // 2) + 1: 3 * (i // 2) + 2][0](x_surf2struct)
#
#                 # struct to surface fusion
#                 x_struct2surf = torch.cat([x_struct, self.padding_struct2surf[i // 2]], dim = 0)
#                 x_struct2surf = x_struct2surf[input['struct2surf_list'][i // 2]]
#                 x_struct2surf = self.struct2surf_blocks[3 * (i // 2) + 0: 3 * (i // 2) + 1][0](x_struct2surf).max(-2)[0]
#                 x_struct2surf = torch.cat([x_surf, x_struct2surf], dim=-1)
#                 x_struct2surf = self.struct2surf_blocks[3 * (i // 2) + 1: 3 * (i // 2) + 2][0](x_struct2surf)
#
#
#                 # update fusion results
#                 x_struct = self.surf2struct_blocks[3 * (i // 2) + 2: 3 * (i // 2) + 3][0](x_struct) + x_surf2struct
#                 x_surf = self.struct2surf_blocks[3 * (i // 2) + 2: 3 * (i // 2) + 3][0](x_surf) + x_struct2surf
#
#
#         output = x_struct
#         out = self.classifier(output)
#
#         return out



# # sparse + distance weight
# # class ProteinF3S_Surf2Struct_MSF_Func(nn.Module):
# #     def __init__(self,
# #                  cfg,
# #                  embedding_dim: int = 16,
# #                  batch_norm: bool = True,
# #                  dropout: float = 0.2,
# #                  bias: bool = False,
# #                  num_classes: int = 384) -> nn.Module:
# #
# #         super().__init__()
# #
# #         self.structure = cfg.structure
# #         self.surface = cfg.surface
# #         self.sequence = cfg.sequence
# #         assert (self.structure & self.surface & (not self.sequence))
# #
# #         geometric_radii = [2 * cfg.geometric_radius, 3 * cfg.geometric_radius, 4 * cfg.geometric_radius, 5 * cfg.geometric_radius]
# #         sequential_kernel_size = cfg.sequential_kernel_size
# #         kernel_channels = cfg.kernel_channels
# #         channels = cfg.channels
# #         base_width = cfg.base_width
# #
# #         assert (len(geometric_radii) == len(
# #             channels)), "Model: 'geometric_radii' and 'channels' should have the same number of elements!"
# #
# #         # fusion
# #         self.surf2struct_blocks = nn.ModuleList()
# #         self.struct2surf_blocks = nn.ModuleList()
# #
# #         # struct
# #         self.embedding = torch.nn.Embedding(num_embeddings=21, embedding_dim=embedding_dim)
# #         self.local_mean_pool = AvgPooling()
# #
# #         layers = []
# #         self.padding_surf2struct = []
# #         self.padding_struct2surf = []
# #
# #         in_channels = embedding_dim
# #         for i, radius in enumerate(geometric_radii):
# #             layers.append(BasicBlock(r=radius,
# #                                      l=sequential_kernel_size,
# #                                      kernel_channels=kernel_channels,
# #                                      in_channels=in_channels,
# #                                      out_channels=channels[i],
# #                                      base_width=base_width,
# #                                      batch_norm=batch_norm,
# #                                      dropout=dropout,
# #                                      bias=bias))
# #             layers.append(BasicBlock(r=radius,
# #                                      l=sequential_kernel_size,
# #                                      kernel_channels=kernel_channels,
# #                                      in_channels=channels[i],
# #                                      out_channels=channels[i],
# #                                      base_width=base_width,
# #                                      batch_norm=batch_norm,
# #                                      dropout=dropout,
# #                                      bias=bias))
# #             in_channels = channels[i]
# #
# #             self.surf2struct_blocks.append(
# #                 Fusion_Block(
# #                     in_channels, radius
# #             ))
# #             self.surf2struct_blocks.append(
# #                 nn.Sequential(
# #                     nn.Linear(in_channels * 2, in_channels, 1),
# #                     nn.BatchNorm1d(in_channels),
# #                     nn.LeakyReLU(0.1),
# #                     nn.Linear(in_channels, in_channels, 1)
# #                 )
# #             )
# #             self.surf2struct_blocks.append(nn.Linear(in_channels, in_channels))
# #
# #
# #             self.struct2surf_blocks.append(
# #                 Fusion_Block(
# #                     in_channels, radius
# #                 )
# #             )
# #             self.struct2surf_blocks.append(
# #                 nn.Sequential(
# #                     nn.Linear(in_channels * 2, in_channels, 1),
# #                     nn.BatchNorm1d(in_channels),
# #                     nn.LeakyReLU(0.1),
# #                     nn.Linear(in_channels, in_channels, 1)
# #                 )
# #             )
# #             self.struct2surf_blocks.append(nn.Linear(in_channels, in_channels))
# #
# #             self.padding_surf2struct.append(
# #                 nn.init.xavier_uniform_(nn.Parameter(torch.zeros(1, in_channels))).cuda()
# #             )
# #             self.padding_struct2surf.append(
# #                 nn.init.xavier_uniform_(nn.Parameter(torch.zeros(1, in_channels))).cuda()
# #             )
# #
# #         self.layers = nn.Sequential(*layers)
# #
# #         # surface
# #         self.use_chem = cfg.use_chem
# #         self.use_geo = cfg.use_geo
# #
# #         in_feats_dim = 1
# #
# #         if self.use_chem:
# #             self.mini_pointnet = nn.ModuleList()
# #             self.mini_pointnet.append(nn.Linear(15, 21))
# #             self.mini_pointnet.append(nn.BatchNorm1d(21))
# #             self.mini_pointnet.append(nn.LeakyReLU(0.1))
# #             self.mini_pointnet.append(nn.Linear(21, 21))
# #             in_feats_dim += 21
# #         if self.use_geo:
# #             in_feats_dim += 10
# #
# #         self.KPFCN = KPFCN(cfg)
# #
# #         self.classifier = MLP(in_channels=2 * channels[-1],
# #                               mid_channels=max(channels[-1], num_classes),
# #                               out_channels=num_classes,
# #                               batch_norm=batch_norm,
# #                               dropout=dropout)
# #
# #
# #     def forward(self, input):
# #
# #         feat = input['features'].clone().detach()
# #         if self.use_chem:
# #             chem = input['chem']
# #             for i, layer in enumerate(self.mini_pointnet):
# #                 if i == 1:
# #                     chem = chem.transpose(2, 1)
# #                     chem = layer(chem)
# #                     chem = chem.transpose(2, 1)
# #                 else:
# #                     chem = layer(chem)
# #             chem = chem.max(-2)[0]
# #             feat = torch.cat([chem, feat], dim=-1)
# #         if self.use_geo:
# #             geo = input['geo']
# #             feat = torch.cat([geo, feat], dim=-1)
# #         x_surf = feat
# #
# #         x = input['x']
# #         pos = input['pos']
# #         seq = input['seq']
# #         ori = input['ori']
# #         batch = input['batch']
# #         x_struct, pos, seq, ori, batch = (self.embedding(x), pos, seq, ori, batch)
# #
# #
# #         for i, layer in enumerate(self.layers):
# #             x_struct = layer(x_struct, pos, seq, ori, batch)
# #             if i == len(self.layers) - 1:
# #                 x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 0: 3 * (i // 2) + 1][0](x_surf, input)
# #                 x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 1: 3 * (i // 2) + 2][0](x_surf, input)
# #                 x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 2: 3 * (i // 2) + 3][0](x_surf, input)
# #
# #                 x_struct = global_mean_pool(x_struct, batch)
# #
# #                 x_struct = torch.cat([x_struct, x_surf], dim=-1)
# #             elif i % 2 == 1:
# #
# #                 x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 0: 3 * (i // 2) + 1][0](x_surf, input)
# #                 x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 1: 3 * (i // 2) + 2][0](x_surf, input)
# #
# #                 # downsample
# #                 x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 2: 3 * (i // 2) + 3][0](x_surf, input)
# #                 x_struct, pos, seq, ori, batch = self.local_mean_pool(x_struct, pos, seq, ori, batch)
# #
# #                 # surface to struct fusion
# #                 x_surf2struct = self.surf2struct_blocks[3 * (i // 2) + 0: 3 * (i // 2) + 1][0](x_surf,
# #                                                                 self.padding_surf2struct[i // 2],
# #                                                                 input['surf2struct_list'][i // 2],
# #                                                                 input['points'][i // 2 + 1],
# #                                                                 pos)
# #                 x_surf2struct = torch.cat([x_struct, x_surf2struct], dim = -1)
# #                 x_surf2struct = self.surf2struct_blocks[3 * (i // 2) + 1: 3 * (i // 2) + 2][0](x_surf2struct)
# #
# #                 # struct to surface fusion
# #                 x_struct2surf = self.struct2surf_blocks[3 * (i // 2) + 0: 3 * (i // 2) + 1][0](x_struct,
# #                                                                 self.padding_struct2surf[i // 2],
# #                                                                 input['struct2surf_list'][i // 2],
# #                                                                 pos,
# #                                                                 input['points'][i // 2 + 1])
# #                 x_struct2surf = torch.cat([x_surf, x_struct2surf], dim=-1)
# #                 x_struct2surf = self.struct2surf_blocks[3 * (i // 2) + 1: 3 * (i // 2) + 2][0](x_struct2surf)
# #
# #
# #                 # update fusion results
# #                 x_struct = self.surf2struct_blocks[3 * (i // 2) + 2: 3 * (i // 2) + 3][0](x_struct) + x_surf2struct
# #                 x_surf = self.struct2surf_blocks[3 * (i // 2) + 2: 3 * (i // 2) + 3][0](x_surf) + x_struct2surf
# #
# #
# #         output = x_struct
# #         out = self.classifier(output)
# #
# #         return out
#
#
#
# class ProteinF3S_Surf2Struct_MSF_Func(nn.Module):
#     def __init__(self,
#                  cfg,
#                  embedding_dim: int = 16,
#                  batch_norm: bool = True,
#                  dropout: float = 0.2,
#                  bias: bool = False,
#                  num_classes: int = 384) -> nn.Module:
#
#         super().__init__()
#
#         self.structure = cfg.structure
#         self.surface = cfg.surface
#         self.sequence = cfg.sequence
#         assert (self.structure & self.surface & (not self.sequence))
#
#         geometric_radii = [2 * cfg.geometric_radius, 3 * cfg.geometric_radius, 4 * cfg.geometric_radius, 5 * cfg.geometric_radius]
#         sequential_kernel_size = cfg.sequential_kernel_size
#         kernel_channels = cfg.kernel_channels
#         channels = cfg.channels
#         base_width = cfg.base_width
#
#         assert (len(geometric_radii) == len(
#             channels)), "Model: 'geometric_radii' and 'channels' should have the same number of elements!"
#
#         # fusion
#         self.surf2struct_blocks = nn.ModuleList()
#         self.struct2surf_blocks = nn.ModuleList()
#
#         # struct
#         self.embedding = torch.nn.Embedding(num_embeddings=21, embedding_dim=embedding_dim)
#         self.local_mean_pool = AvgPooling()
#
#         layers = []
#         self.padding_surf2struct = []
#         self.padding_struct2surf = []
#
#         in_channels = embedding_dim
#         for i, radius in enumerate(geometric_radii):
#             layers.append(BasicBlock(r=radius,
#                                      l=sequential_kernel_size,
#                                      kernel_channels=kernel_channels,
#                                      in_channels=in_channels,
#                                      out_channels=channels[i],
#                                      base_width=base_width,
#                                      batch_norm=batch_norm,
#                                      dropout=dropout,
#                                      bias=bias))
#             layers.append(BasicBlock(r=radius,
#                                      l=sequential_kernel_size,
#                                      kernel_channels=kernel_channels,
#                                      in_channels=channels[i],
#                                      out_channels=channels[i],
#                                      base_width=base_width,
#                                      batch_norm=batch_norm,
#                                      dropout=dropout,
#                                      bias=bias))
#             in_channels = channels[i]
#
#
#             self.surf2struct_blocks.append(
#                 nn.Sequential(
#                     nn.Linear(in_channels, in_channels),
#                     nn.LayerNorm(in_channels),
#                     nn.LeakyReLU(0.1),
#             ))
#             self.surf2struct_blocks.append(
#                 nn.Sequential(
#                     nn.Linear(in_channels * 2, in_channels, 1),
#                     nn.BatchNorm1d(in_channels),
#                     nn.LeakyReLU(0.1),
#                     nn.Linear(in_channels, in_channels, 1)
#                 )
#             )
#             self.surf2struct_blocks.append(nn.Linear(in_channels, in_channels))
#
#
#             self.struct2surf_blocks.append(
#                 nn.Sequential(
#                     nn.Linear(in_channels, in_channels),
#                     nn.LayerNorm(in_channels),
#                     nn.LeakyReLU(0.1),
#                 )
#             )
#             self.struct2surf_blocks.append(
#                 nn.Sequential(
#                     nn.Linear(in_channels * 2, in_channels, 1),
#                     nn.BatchNorm1d(in_channels),
#                     nn.LeakyReLU(0.1),
#                     nn.Linear(in_channels, in_channels, 1)
#                 )
#             )
#             self.struct2surf_blocks.append(nn.Linear(in_channels, in_channels))
#
#             self.padding_surf2struct.append(
#                 nn.init.xavier_uniform_(nn.Parameter(torch.zeros(1, in_channels))).cuda()
#             )
#             self.padding_struct2surf.append(
#                 nn.init.xavier_uniform_(nn.Parameter(torch.zeros(1, in_channels))).cuda()
#             )
#
#
#
#         self.layers = nn.Sequential(*layers)
#
#         # surface
#         self.use_chem = cfg.use_chem
#         self.use_geo = cfg.use_geo
#
#         in_feats_dim = 1
#
#         if self.use_chem:
#             self.mini_pointnet = nn.ModuleList()
#             self.mini_pointnet.append(nn.Linear(15, 21))
#             self.mini_pointnet.append(nn.BatchNorm1d(21))
#             self.mini_pointnet.append(nn.LeakyReLU(0.1))
#             self.mini_pointnet.append(nn.Linear(21, 21))
#             in_feats_dim += 21
#         if self.use_geo:
#             in_feats_dim += 10
#
#         self.KPFCN = KPFCN(cfg)
#
#         self.classifier = MLP(in_channels=channels[-1],
#                               mid_channels=max(channels[-1], num_classes),
#                               out_channels=num_classes,
#                               batch_norm=batch_norm,
#                               dropout=dropout)
#
#
#     def forward(self, input):
#
#         feat = input['features'].clone().detach()
#         if self.use_chem:
#             chem = input['chem']
#             for i, layer in enumerate(self.mini_pointnet):
#                 if i == 1:
#                     chem = chem.transpose(2, 1)
#                     chem = layer(chem)
#                     chem = chem.transpose(2, 1)
#                 else:
#                     chem = layer(chem)
#             chem = chem.max(-2)[0]
#             feat = torch.cat([chem, feat], dim=-1)
#         if self.use_geo:
#             geo = input['geo']
#             feat = torch.cat([geo, feat], dim=-1)
#         x_surf = feat
#
#         x = input['x']
#         pos = input['pos']
#         seq = input['seq']
#         ori = input['ori']
#         batch = input['batch']
#         x_struct, pos, seq, ori, batch = (self.embedding(x), pos, seq, ori, batch)
#
#
#         for i, layer in enumerate(self.layers):
#             x_struct = layer(x_struct, pos, seq, ori, batch)
#             if i == len(self.layers) - 1:
#                 x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 0: 3 * (i // 2) + 1][0](x_surf, input)
#                 x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 1: 3 * (i // 2) + 2][0](x_surf, input)
#
#                 x_surf2struct = torch.cat([x_surf, self.padding_surf2struct[i // 2]], dim=0)[input['surf2struct_list'][i // 2]]
#                 x_surf2struct = self.surf2struct_blocks[3 * (i // 2) + 0: 3 * (i // 2) + 1][0](
#                     x_surf2struct).mean(-2)
#                 x_surf2struct = torch.cat([x_struct, x_surf2struct], dim=-1)
#                 x_surf2struct = self.surf2struct_blocks[3 * (i // 2) + 1: 3 * (i // 2) + 2][0](x_surf2struct)
#                 x_struct = self.surf2struct_blocks[3 * (i // 2) + 2: 3 * (i // 2) + 3][0](x_struct) + x_surf2struct
#
#                 x_struct = global_mean_pool(x_struct, batch)
#             elif i % 2 == 1:
#
#                 x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 0: 3 * (i // 2) + 1][0](x_surf, input)
#                 x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 1: 3 * (i // 2) + 2][0](x_surf, input)
#
#                 # surface to struct fusion
#                 x_surf2struct = torch.cat([x_surf, self.padding_surf2struct[i // 2]], dim=0)[input['surf2struct_list'][i // 2]]
#                 x_surf2struct = self.surf2struct_blocks[3 * (i // 2) + 0: 3 * (i // 2) + 1][0](x_surf2struct).max(-2)[0]
#                 x_surf2struct = torch.cat([x_struct, x_surf2struct], dim = -1)
#                 x_surf2struct = self.surf2struct_blocks[3 * (i // 2) + 1: 3 * (i // 2) + 2][0](x_surf2struct)
#
#                 # struct to surface fusion
#                 x_struct2surf = torch.cat([x_struct, self.padding_struct2surf[i // 2]], dim=0)[input['struct2surf_list'][i // 2]]
#                 x_struct2surf = self.struct2surf_blocks[3 * (i // 2) + 0: 3 * (i // 2) + 1][0](x_struct2surf).max(-2)[0]
#                 x_struct2surf = torch.cat([x_surf, x_struct2surf], dim=-1)
#                 x_struct2surf = self.struct2surf_blocks[3 * (i // 2) + 1: 3 * (i // 2) + 2][0](x_struct2surf)
#
#
#                 # update fusion results
#                 x_struct = self.surf2struct_blocks[3 * (i // 2) + 2: 3 * (i // 2) + 3][0](x_struct) + x_surf2struct
#                 x_surf = self.struct2surf_blocks[3 * (i // 2) + 2: 3 * (i // 2) + 3][0](x_surf) + x_struct2surf
#
#
#                 # downsample
#                 x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 2: 3 * (i // 2) + 3][0](x_surf, input)
#                 x_struct, pos, seq, ori, batch = self.local_mean_pool(x_struct, pos, seq, ori, batch)
#
#
#         output = x_struct
#         out = self.classifier(output)
#
#         return out



# x_surf2struct = self.surf2struct_blocks[3 * (i // 2) + 0: 3 * (i // 2) + 1][0](x_surf,
#                                                                 self.padding_surf2struct[i // 2],
#                                                                 input['surf2struct_list'][i // 2],
#                                                                 input['points'][i // 2 + 1],
#                                                                 pos)

# sparse
class ProteinF3S_Surf2Struct_MSF_Func(nn.Module):
    def __init__(self,
                 cfg,
                 embedding_dim: int = 16,
                 batch_norm: bool = True,
                 dropout: float = 0.2,
                 bias: bool = False,
                 num_classes: int = 384) -> nn.Module:
        super().__init__()
        self.structure = cfg.structure
        self.surface = cfg.surface
        self.sequence = cfg.sequence
        assert (self.structure & self.surface & (not self.sequence))
        geometric_radii = [2 * cfg.geometric_radius, 3 * cfg.geometric_radius, 4 * cfg.geometric_radius, 5 * cfg.geometric_radius]
        sequential_kernel_size = cfg.sequential_kernel_size
        kernel_channels = cfg.kernel_channels
        channels = cfg.channels
        base_width = cfg.base_width
        assert (len(geometric_radii) == len(
            channels)), "Model: 'geometric_radii' and 'channels' should have the same number of elements!"
        # fusion
        self.surf2struct_blocks = nn.ModuleList()
        self.struct2surf_blocks = nn.ModuleList()
        # struct
        self.embedding = torch.nn.Embedding(num_embeddings=21, embedding_dim=embedding_dim)
        self.local_mean_pool = AvgPooling()
        layers = []
        self.padding_surf2struct = []
        self.padding_struct2surf = []
        in_channels = embedding_dim
        for i, radius in enumerate(geometric_radii):
            layers.append(BasicBlock(r=radius,
                                     l=sequential_kernel_size,
                                     kernel_channels=kernel_channels,
                                     in_channels=in_channels,
                                     out_channels=channels[i],
                                     base_width=base_width,
                                     batch_norm=batch_norm,
                                     dropout=dropout,
                                     bias=bias))
            layers.append(BasicBlock(r=radius,
                                     l=sequential_kernel_size,
                                     kernel_channels=kernel_channels,
                                     in_channels=channels[i],
                                     out_channels=channels[i],
                                     base_width=base_width,
                                     batch_norm=batch_norm,
                                     dropout=dropout,
                                     bias=bias))
            in_channels = channels[i]
            self.surf2struct_blocks.append(
                nn.Sequential(
                    nn.Linear(in_channels, in_channels),
                    nn.LayerNorm(in_channels),
                    nn.LeakyReLU(0.1),
            ))
            self.surf2struct_blocks.append(
                nn.Sequential(
                    nn.Linear(in_channels * 2, in_channels, 1),
                    nn.BatchNorm1d(in_channels),
                    nn.LeakyReLU(0.1),
                    nn.Linear(in_channels, in_channels, 1)
                )
            )
            self.surf2struct_blocks.append(nn.Linear(in_channels, in_channels))
            self.struct2surf_blocks.append(
                nn.Sequential(
                    nn.Linear(in_channels, in_channels),
                    nn.LayerNorm(in_channels),
                    nn.LeakyReLU(0.1),
                )
            )
            self.struct2surf_blocks.append(
                nn.Sequential(
                    nn.Linear(in_channels * 2, in_channels, 1),
                    nn.BatchNorm1d(in_channels),
                    nn.LeakyReLU(0.1),
                    nn.Linear(in_channels, in_channels, 1)
                )
            )
            self.struct2surf_blocks.append(nn.Linear(in_channels, in_channels))
            self.padding_surf2struct.append(
                nn.init.xavier_uniform_(nn.Parameter(torch.zeros(1, in_channels))).cuda()
            )
            self.padding_struct2surf.append(
                nn.init.xavier_uniform_(nn.Parameter(torch.zeros(1, in_channels))).cuda()
            )
        self.layers = nn.Sequential(*layers)
        # surface
        self.use_chem = cfg.use_chem
        self.use_geo = cfg.use_geo
        in_feats_dim = 1
        if self.use_chem:
            self.mini_pointnet = nn.ModuleList()
            self.mini_pointnet.append(nn.Linear(15, 21))
            self.mini_pointnet.append(nn.BatchNorm1d(21))
            self.mini_pointnet.append(nn.LeakyReLU(0.1))
            self.mini_pointnet.append(nn.Linear(21, 21))
            in_feats_dim += 21
        if self.use_geo:
            in_feats_dim += 10
        self.KPFCN = KPFCN(cfg)
        self.classifier = MLP(in_channels=1 * channels[-1],
                              mid_channels=max(channels[-1], num_classes),
                              out_channels=num_classes,
                              batch_norm=batch_norm,
                              dropout=dropout)
    def forward(self, input):
        feat = input['features'].clone().detach()
        if self.use_chem:
            chem = input['chem']
            for i, layer in enumerate(self.mini_pointnet):
                if i == 1:
                    chem = chem.transpose(2, 1)
                    chem = layer(chem)
                    chem = chem.transpose(2, 1)
                else:
                    chem = layer(chem)
            chem = chem.max(-2)[0]
            feat = torch.cat([chem, feat], dim=-1)
        if self.use_geo:
            geo = input['geo']
            feat = torch.cat([geo, feat], dim=-1)
        x_surf = feat
        x = input['x']
        pos = input['pos']
        seq = input['seq']
        ori = input['ori']
        batch = input['batch']
        x_struct, pos, seq, ori, batch = (self.embedding(x), pos, seq, ori, batch)
        for i, layer in enumerate(self.layers):
            x_struct = layer(x_struct, pos, seq, ori, batch)
            if i == len(self.layers) - 1:
                # x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 0: 3 * (i // 2) + 1][0](x_surf, input)
                # x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 1: 3 * (i // 2) + 2][0](x_surf, input)
                # x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 2: 3 * (i // 2) + 3][0](x_surf, input)
                x_struct = global_mean_pool(x_struct, batch)
                # x_struct = torch.cat([x_struct, x_surf], dim=-1)
                # x_struct = x_surf
            elif i % 2 == 1:
                x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 0: 3 * (i // 2) + 1][0](x_surf, input)
                x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 1: 3 * (i // 2) + 2][0](x_surf, input)
                # downsample
                x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 2: 3 * (i // 2) + 3][0](x_surf, input)
                x_struct, pos, seq, ori, batch = self.local_mean_pool(x_struct, pos, seq, ori, batch)
                # surface to struct fusion
                x_surf2struct = torch.cat([x_surf, self.padding_surf2struct[i // 2]], dim = 0)
                x_struct2surf = x_surf2struct[input['surf2struct_list'][i // 2]]
                x_surf2struct = self.surf2struct_blocks[3 * (i // 2) + 0: 3 * (i // 2) + 1][0](x_struct2surf).max(-2)[0]
                x_surf2struct = torch.cat([x_struct, x_surf2struct], dim = -1)
                x_surf2struct = self.surf2struct_blocks[3 * (i // 2) + 1: 3 * (i // 2) + 2][0](x_surf2struct)
                # struct to surface fusion
                x_struct2surf = torch.cat([x_struct, self.padding_struct2surf[i // 2]], dim = 0)
                x_struct2surf = x_struct2surf[input['struct2surf_list'][i // 2]]
                x_struct2surf = self.struct2surf_blocks[3 * (i // 2) + 0: 3 * (i // 2) + 1][0](x_struct2surf).max(-2)[0]
                x_struct2surf = torch.cat([x_surf, x_struct2surf], dim=-1)
                x_struct2surf = self.struct2surf_blocks[3 * (i // 2) + 1: 3 * (i // 2) + 2][0](x_struct2surf)
                # update fusion results
                x_struct = self.surf2struct_blocks[3 * (i // 2) + 2: 3 * (i // 2) + 3][0](x_struct) + x_surf2struct
                x_surf = self.struct2surf_blocks[3 * (i // 2) + 2: 3 * (i // 2) + 3][0](x_surf) + x_struct2surf
        output = x_struct
        out = self.classifier(output)
        return out


class ProteinF3S(nn.Module):
    def __init__(self,
                 cfg,
                 embedding_dim: int = 16,
                 batch_norm: bool = True,
                 dropout: float = 0.2,
                 bias: bool = False,
                 num_classes: int = 384) -> nn.Module:
        super().__init__()
        self.structure = cfg.structure
        self.surface = cfg.surface
        self.sequence = cfg.sequence
        assert (self.structure & self.surface & self.sequence)
        geometric_radii = [2 * cfg.geometric_radius, 3 * cfg.geometric_radius, 4 * cfg.geometric_radius, 5 * cfg.geometric_radius]
        sequential_kernel_size = cfg.sequential_kernel_size
        kernel_channels = cfg.kernel_channels
        channels = cfg.channels
        base_width = cfg.base_width
        assert (len(geometric_radii) == len(
            channels)), "Model: 'geometric_radii' and 'channels' should have the same number of elements!"
        # fusion
        self.surf2struct_blocks = nn.ModuleList()
        self.struct2surf_blocks = nn.ModuleList()
        # struct
        self.embedding = torch.nn.Embedding(num_embeddings=21, embedding_dim=embedding_dim)
        self.local_mean_pool = AvgPooling()
        layers = []
        self.padding_surf2struct = []
        self.padding_struct2surf = []
        in_channels = embedding_dim
        for i, radius in enumerate(geometric_radii):
            layers.append(BasicBlock(r=radius,
                                     l=sequential_kernel_size,
                                     kernel_channels=kernel_channels,
                                     in_channels=in_channels,
                                     out_channels=channels[i],
                                     base_width=base_width,
                                     batch_norm=batch_norm,
                                     dropout=dropout,
                                     bias=bias))
            layers.append(BasicBlock(r=radius,
                                     l=sequential_kernel_size,
                                     kernel_channels=kernel_channels,
                                     in_channels=channels[i],
                                     out_channels=channels[i],
                                     base_width=base_width,
                                     batch_norm=batch_norm,
                                     dropout=dropout,
                                     bias=bias))
            in_channels = channels[i]
            self.surf2struct_blocks.append(
                nn.Sequential(
                    nn.Linear(in_channels, in_channels),
                    nn.LayerNorm(in_channels),
                    nn.LeakyReLU(0.1),
            ))
            self.surf2struct_blocks.append(
                nn.Sequential(
                    nn.Linear(in_channels * 2, in_channels, 1),
                    nn.BatchNorm1d(in_channels),
                    nn.LeakyReLU(0.1),
                    nn.Linear(in_channels, in_channels, 1)
                )
            )
            self.surf2struct_blocks.append(nn.Linear(in_channels, in_channels))
            self.struct2surf_blocks.append(
                nn.Sequential(
                    nn.Linear(in_channels, in_channels),
                    nn.LayerNorm(in_channels),
                    nn.LeakyReLU(0.1),
                )
            )
            self.struct2surf_blocks.append(
                nn.Sequential(
                    nn.Linear(in_channels * 2, in_channels, 1),
                    nn.BatchNorm1d(in_channels),
                    nn.LeakyReLU(0.1),
                    nn.Linear(in_channels, in_channels, 1)
                )
            )
            self.struct2surf_blocks.append(nn.Linear(in_channels, in_channels))
            self.padding_surf2struct.append(
                nn.init.xavier_uniform_(nn.Parameter(torch.zeros(1, in_channels))).cuda()
            )
            self.padding_struct2surf.append(
                nn.init.xavier_uniform_(nn.Parameter(torch.zeros(1, in_channels))).cuda()
            )
        self.layers = nn.Sequential(*layers)
        # surface
        self.use_chem = cfg.use_chem
        self.use_geo = cfg.use_geo
        in_feats_dim = 1
        if self.use_chem:
            self.mini_pointnet = nn.ModuleList()
            self.mini_pointnet.append(nn.Linear(15, 21))
            self.mini_pointnet.append(nn.BatchNorm1d(21))
            self.mini_pointnet.append(nn.LeakyReLU(0.1))
            self.mini_pointnet.append(nn.Linear(21, 21))
            in_feats_dim += 21
        if self.use_geo:
            in_feats_dim += 10
        self.KPFCN = KPFCN(cfg)

        if cfg.capacity == 'tiny':
            model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            self.repr_layers = [6]
            self.output_dim = 320
        elif cfg.capacity == 'normal':
            model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            self.repr_layers = [12]
            self.output_dim = 480
        else:
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.repr_layers = [33]
            self.output_dim = 1280
        self.capacity = cfg.capacity
        self.alphabet = alphabet
        self.sequence_model = model
        self.max_input_length = cfg.max_input_length

        self.tune = cfg.tune
        if self.tune == 'ft':
            pass
        if self.tune == 'lp':
            self.sequence_model.eval()
            for k, v in self.sequence_model.named_parameters():
                v.requires_grad = False

        self.classifier = MLP(in_channels=1 * channels[-1] + self.output_dim,
                              mid_channels=max(channels[-1], num_classes),
                              out_channels=num_classes,
                              batch_norm=batch_norm,
                              dropout=dropout)


    def forward(self, input):
        feat = input['features'].clone().detach()
        if self.use_chem:
            chem = input['chem']
            for i, layer in enumerate(self.mini_pointnet):
                if i == 1:
                    chem = chem.transpose(2, 1)
                    chem = layer(chem)
                    chem = chem.transpose(2, 1)
                else:
                    chem = layer(chem)
            chem = chem.max(-2)[0]
            feat = torch.cat([chem, feat], dim=-1)
        if self.use_geo:
            geo = input['geo']
            feat = torch.cat([geo, feat], dim=-1)
        x_surf = feat
        x = input['x']
        pos = input['pos']
        seq = input['seq']
        ori = input['ori']
        batch = input['batch']
        x_struct, pos, seq, ori, batch = (self.embedding(x), pos, seq, ori, batch)
        for i, layer in enumerate(self.layers):
            x_struct = layer(x_struct, pos, seq, ori, batch)
            if i == len(self.layers) - 1:
                # x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 0: 3 * (i // 2) + 1][0](x_surf, input)
                # x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 1: 3 * (i // 2) + 2][0](x_surf, input)
                # x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 2: 3 * (i // 2) + 3][0](x_surf, input)
                x_struct = global_mean_pool(x_struct, batch)
                # x_struct = torch.cat([x_struct, x_surf], dim=-1)
                # x_struct = x_surf
            elif i % 2 == 1:
                x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 0: 3 * (i // 2) + 1][0](x_surf, input)
                x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 1: 3 * (i // 2) + 2][0](x_surf, input)
                # downsample
                x_surf = self.KPFCN.encoder_blocks[3 * (i // 2) + 2: 3 * (i // 2) + 3][0](x_surf, input)
                x_struct, pos, seq, ori, batch = self.local_mean_pool(x_struct, pos, seq, ori, batch)
                # surface to struct fusion
                x_surf2struct = torch.cat([x_surf, self.padding_surf2struct[i // 2]], dim = 0)
                x_struct2surf = x_surf2struct[input['surf2struct_list'][i // 2]]
                x_surf2struct = self.surf2struct_blocks[3 * (i // 2) + 0: 3 * (i // 2) + 1][0](x_struct2surf).max(-2)[0]
                x_surf2struct = torch.cat([x_struct, x_surf2struct], dim = -1)
                x_surf2struct = self.surf2struct_blocks[3 * (i // 2) + 1: 3 * (i // 2) + 2][0](x_surf2struct)
                # struct to surface fusion
                x_struct2surf = torch.cat([x_struct, self.padding_struct2surf[i // 2]], dim = 0)
                x_struct2surf = x_struct2surf[input['struct2surf_list'][i // 2]]
                x_struct2surf = self.struct2surf_blocks[3 * (i // 2) + 0: 3 * (i // 2) + 1][0](x_struct2surf).max(-2)[0]
                x_struct2surf = torch.cat([x_surf, x_struct2surf], dim=-1)
                x_struct2surf = self.struct2surf_blocks[3 * (i // 2) + 1: 3 * (i // 2) + 2][0](x_struct2surf)
                # update fusion results
                x_struct = self.surf2struct_blocks[3 * (i // 2) + 2: 3 * (i // 2) + 3][0](x_struct) + x_surf2struct
                x_surf = self.struct2surf_blocks[3 * (i // 2) + 2: 3 * (i // 2) + 3][0](x_surf) + x_struct2surf
        # output = x_struct

        batch_tokens = input['tokens']
        size = input['token_lens']

        if size.max() > self.max_input_length:
            batch_tokens = batch_tokens[:, :self.max_input_length]
            size = torch.ones_like(size) * self.max_input_length

        results = self.sequence_model(batch_tokens, repr_layers=self.repr_layers, return_contacts=False)
        token_representations = results["representations"][self.repr_layers[-1]]
        node_feature = token_representations

        global_feature = []
        index = []

        for i, i_size in enumerate(size):
            global_feature.append(token_representations[i, 1:i_size - 1])
            index.append((i * torch.ones(i_size - 2, device=node_feature.device, dtype=torch.int64)))

        global_feature = torch.cat(global_feature)
        index = torch.cat(index)

        output_sequence = scatter_mean(global_feature, index, dim=0, dim_size=len(node_feature))

        out = self.classifier(torch.cat([x_struct,output_sequence], dim=-1))

        return out

