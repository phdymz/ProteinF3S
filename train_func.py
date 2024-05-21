import os
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torchvision
import time
from datasets.func import FuncDataset
from datasets.builder import get_dataloader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models.build_models import build_model


def train(epoch, dataloader):
    model.train()
    for data in tqdm(dataloader):
        for k, v in data.items():
            if type(v) == list:
                data[k] = [item.to(device) for item in v]
            elif type(v) in [dict, float, type(None), np.ndarray]:
                pass
            else:
                data[k] = v.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data).log_softmax(dim=-1), data['label'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

def test(dataloader):
    model.eval()
    correct = 0
    for data in tqdm(dataloader):
        for k, v in data.items():
            if type(v) == list:
                data[k] = [item.to(device) for item in v]
            elif type(v) in [dict, float, type(None), np.ndarray]:
                pass
            else:
                data[k] = v.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data['label']).sum().item()
    return correct / len(dataloader.dataset)

def train_GA(epoch, dataloader, acc_iter = 2):
    model.train()
    for batch_idx, data in enumerate(tqdm(dataloader)):
        for k, v in data.items():
            if type(v) == list:
                data[k] = [item.to(device) for item in v]
            elif type(v) in [dict, float, type(None), np.ndarray]:
                pass
            else:
                data[k] = v.to(device)

        loss = F.nll_loss(model(data).log_softmax(dim=-1), data['label']) / acc_iter
        loss.backward()

        if ((batch_idx + 1) % acc_iter == 0) or (batch_idx + 1 == len(dataloader)):
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()


def parse_args():
    architecture = ['simple',
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
                    'global_average']

    import argparse
    parser = argparse.ArgumentParser(description='ProteinF3S')
    parser.add_argument('--model', default='f3s', type=str)
    parser.add_argument('--dataset', default='func', type=str)
    parser.add_argument('--structure', default=True, type=bool)
    parser.add_argument('--surface', default=True, type=bool)
    parser.add_argument('--sequence', default=True, type=bool)


    parser.add_argument('--data_dir', default='/media/ymz/807a6847-34c1-4630-bb53-2a277f79be0c/ProteinF3S/func',
                        type=str, metavar='N', help='data root directory')
    parser.add_argument('--num_epochs', default=400, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--batch_size', default=8, type=int, metavar='N', help='batch size')
    parser.add_argument('--lr', default=0.001, type=float, metavar='N', help='learning rate')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)', dest='weight_decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--lr-milestones', nargs='+', default=[100, 300], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--ckpt_path', default='/media/ymz/807a6847-34c1-4630-bb53-2a277f79be0c/ProteinF3S/func/log/checkpoint/best.pkl',
                        type=str, help='path where to save checkpoint')
    parser.add_argument('--tensorboard-path', default='/media/ymz/807a6847-34c1-4630-bb53-2a277f79be0c/ProteinF3S/func/log/tensorboard',
                        type=str, help='path where to save checkpoint')
    parser.add_argument('--input_type', default='surface_structure_sequence', type=str,
                        choices=['surface', 'surface_structure', 'surface_structure_sequence'])
    parser.add_argument('--acc_iter', default=1, type=int, metavar='N', help='number of grad acc iter')

    # surface
    parser.add_argument('--architectures', default=architecture, type=list)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--deform_radius', type=float, default=5.0)
    parser.add_argument('--first_subsampling_dl', type=float, default=1.5)
    parser.add_argument('--in_feats_dim', type=int, default=32)
    parser.add_argument('--conv_radius', type=float, default=2.5)
    parser.add_argument('--num_kernel_points', type=int, default=15)
    parser.add_argument('--KP_extent', type=float, default=1.2)
    parser.add_argument('--KP_influence', type=str, default='linear')
    parser.add_argument('--aggregation_mode', type=str, default='sum')
    parser.add_argument('--fixed_kernel_points', type=str, default='center')
    parser.add_argument('--use_batch_norm', type=bool, default=True)
    parser.add_argument('--deformable', type=bool, default=False)
    parser.add_argument('--batch_norm_momentum', type=float, default=0.02)
    parser.add_argument('--use_padding', type=bool, default=True)
    parser.add_argument('--first_feats_dim', type=int, default=256)
    parser.add_argument('--in_points_dim', type=int, default=3)
    parser.add_argument('--modulated', type=bool, default=False)
    parser.add_argument('--num_class', type=int, default=384)
    parser.add_argument('--use_chem', type=bool, default=True)
    parser.add_argument('--use_geo', type=bool, default=True)

    # sequence
    parser.add_argument('--capacity', default='huge', type=str)
    parser.add_argument('--tune', default='lp', type=str, choices=['ft', 'lp', 'lora'])
    parser.add_argument('--max_input_length', default=1024, type=int)

    # structure
    parser.add_argument('--geometric-radius', default=4.0, type=float, metavar='N', help='initial 3D ball query radius')
    parser.add_argument('--sequential-kernel-size', default=21, type=int, metavar='N', help='1D sequential kernel size')
    parser.add_argument('--kernel-channels', default=[24], type=int, metavar='N', help='kernel channels')
    parser.add_argument('--base-width', default=32, type=float, metavar='N', help='bottleneck width')
    parser.add_argument('--channels', nargs='+', default=[256, 512, 1024, 2048], type=int, metavar='N',
                        help='feature channels')

    # Fusion
    parser.add_argument('--use_res', type=bool, default=True)
    parser.add_argument('--fusion_type', type=str, default= 'msf', choices=['cascade', 'cat', 'msf'])
    parser.add_argument('--surface2struct', type=bool, default=True)
    parser.add_argument('--use_superpoint', type=bool, default=True)
    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--scale', type=float, default=0.8)
    parser.add_argument('--K_surf2struct', type=int, default=16)
    parser.add_argument('--K_struct2surf', type=int, default=1)
    parser.add_argument('--use_dense', type=bool, default=False)



    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = FuncDataset(args, root=args.data_dir, random_seed=args.seed, split='training', rotation = True)
    valid_dataset = FuncDataset(args, root=args.data_dir, random_seed=args.seed, split='validation', rotation = False)
    test_dataset = FuncDataset(args, root=args.data_dir, random_seed=args.seed, split='testing', rotation = False)

    model = build_model(args).to(device)
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=args.weight_decay, lr=args.lr, momentum=args.momentum)

    if args.sequence:
        args.alphabet = model.alphabet
        args.batch_converter = args.alphabet.get_batch_converter()

    train_loader, neighborhood_limits = get_dataloader(train_dataset, 'training', args)
    valid_loader, _ = get_dataloader(valid_dataset, 'validation', args, neighborhood_limits)
    test_loader, _ = get_dataloader(test_dataset, 'testing', args, neighborhood_limits)

    # learning rate scheduler
    # lr_weights = []
    # for i, milestone in enumerate(args.lr_milestones):
    #     if i == 0:
    #         lr_weights += [np.power(args.lr_gamma, i)] * milestone
    #     else:
    #         lr_weights += [np.power(args.lr_gamma, i)] * (milestone - args.lr_milestones[i-1])
    # if args.lr_milestones[-1] < args.num_epochs:
    #     lr_weights += [np.power(args.lr_gamma, len(args.lr_milestones))] * (args.num_epochs + 1 - args.lr_milestones[-1])
    # lambda_lr = lambda epoch: lr_weights[epoch]
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    logger = SummaryWriter(log_dir=os.path.join(args.tensorboard_path, time.strftime("%m%d-%H%M")))

    best_valid_acc = best_test_acc = best_acc = 0.0
    best_valid_acc_history = 0
    best_epoch = 0
    for epoch in range(args.num_epochs):
        if args.acc_iter > 1:
            train_GA(epoch, train_loader, args.acc_iter)
        else:
            train(epoch, train_loader)
        lr_scheduler.step()
        valid_acc = test(valid_loader)
        test_acc = test(test_loader)
        logger.add_scalar("valid_acc", valid_acc, epoch+1)
        logger.add_scalar("test_acc", test_acc, epoch+1)

        print(f'Epoch: {epoch+1:03d}, Validation: {valid_acc:.4f}, Test: {test_acc:.4f}')
        if valid_acc >= best_valid_acc:
            best_acc = test_acc
            best_epoch = epoch
            best_valid_acc = valid_acc
            checkpoint = model.state_dict()
            if args.ckpt_path:
                torch.save(checkpoint, osp.join(args.ckpt_path.replace('best.pkl', 'best_{}.pkl'.format(epoch))))
            best_valid_acc_history = max(test_acc, best_valid_acc_history)

        best_test_acc = max(test_acc, best_test_acc)
        print(f'Best: {best_epoch+1:03d}, Validation: {best_valid_acc:.4f}, Test: {best_test_acc:.4f}, Valided Test: {best_acc:.4f}, Best_valided_Test: {best_valid_acc_history:.4f}')
