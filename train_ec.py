import os
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torchvision
import time
from datasets.ec import ECDataset
from datasets.builder import get_dataloader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models.proteinf3s_func import ProteinF3S


def train(epoch, dataloader, loss_fn):
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
        y = data['label'].to(device)
        loss = loss_fn(model(data).sigmoid(), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()


def test(dataloader):
    model.eval()
    probs = []
    labels = []
    for data in tqdm(dataloader):
        for k, v in data.items():
            if type(v) == list:
                data[k] = [item.to(device) for item in v]
            elif type(v) in [dict, float, type(None), np.ndarray]:
                pass
            else:
                data[k] = v.to(device)
        with torch.no_grad():
            prob = model(data).sigmoid().detach().cpu().numpy()
            y = data['label'].cpu().numpy()
        probs.append(prob)
        labels.append(y)
    probs = np.concatenate(probs, axis=0)
    labels = np.concatenate(labels, axis=0)

    return fmax(probs, labels)

def fmax(probs, labels):
    thresholds = np.arange(0, 1, 0.01)
    f_max = 0.0

    for threshold in thresholds:
        precision = 0.0
        recall = 0.0
        precision_cnt = 0
        recall_cnt = 0
        for idx in range(probs.shape[0]):
            prob = probs[idx]
            label = labels[idx]
            pred = (prob > threshold).astype(np.int32)
            correct_sum = np.sum(label*pred)
            pred_sum = np.sum(pred)
            label_sum = np.sum(label)
            if pred_sum > 0:
                precision += correct_sum/pred_sum
                precision_cnt += 1
            if label_sum > 0:
                recall += correct_sum/label_sum
            recall_cnt += 1
        if recall_cnt > 0:
            recall = recall / recall_cnt
        else:
            recall = 0
        if precision_cnt > 0:
            precision = precision / precision_cnt
        else:
            precision = 0
        f = (2.*precision*recall)/max(precision+recall, 1e-8)
        f_max = max(f, f_max)

    return f_max


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
    parser.add_argument('--dataset', default='ec', type=str)
    parser.add_argument('--structure', default=True, type=bool)
    parser.add_argument('--surface', default=True, type=bool)
    parser.add_argument('--sequence', default=True, type=bool)


    parser.add_argument('--data_dir', default='/media/ymz/807a6847-34c1-4630-bb53-2a277f79be0c/ProteinF3S/ec',
                        type=str, metavar='N', help='data root directory')
    parser.add_argument('--num_epochs', default=500, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--batch_size', default=8, type=int, metavar='N', help='batch size')
    parser.add_argument('--lr', default=0.001, type=float, metavar='N', help='learning rate')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)', dest='weight_decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--lr-milestones', nargs='+', default=[100, 300], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--ckpt_path', default='/media/ymz/807a6847-34c1-4630-bb53-2a277f79be0c/ProteinF3S/ec/log/checkpoint/best.pkl',
                        type=str, help='path where to save checkpoint')
    parser.add_argument('--tensorboard-path', default='/media/ymz/807a6847-34c1-4630-bb53-2a277f79be0c/ProteinF3S/ec/log/tensorboard',
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
    parser.add_argument('--kernel-channels', default=[32], type=int, metavar='N', help='kernel channels')
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

    train_dataset = ECDataset(args, root=args.data_dir, random_seed=args.seed, split='train', rotation = True)
    valid_dataset = ECDataset(args, root=args.data_dir, random_seed=args.seed, split='valid', rotation = False)
    test_dataset_30 = ECDataset(args, root=args.data_dir, percent=30, random_seed=args.seed, split='test', rotation = False)
    test_dataset_40 = ECDataset(args, root=args.data_dir, percent=40, random_seed=args.seed, split='test', rotation=False)
    test_dataset_50 = ECDataset(args, root=args.data_dir, percent=50, random_seed=args.seed, split='test', rotation=False)
    test_dataset_70 = ECDataset(args, root=args.data_dir, percent=70, random_seed=args.seed, split='test', rotation=False)
    test_dataset_95 = ECDataset(args, root=args.data_dir, percent=95, random_seed=args.seed, split='test', rotation=False)


    model = ProteinF3S(args, num_classes=train_dataset.num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=args.weight_decay, lr=args.lr, momentum=args.momentum)
    loss_fn = torch.nn.BCELoss(weight=torch.as_tensor(train_dataset.weights).to(device))

    if args.sequence:
        args.alphabet = model.alphabet
        args.batch_converter = args.alphabet.get_batch_converter()

    train_loader, neighborhood_limits = get_dataloader(train_dataset, 'training', args)
    valid_loader, _ = get_dataloader(valid_dataset, 'validation', args, neighborhood_limits)
    test_loader_30, _ = get_dataloader(test_dataset_30, 'testing', args, neighborhood_limits)
    test_loader_40, _ = get_dataloader(test_dataset_40, 'testing', args, neighborhood_limits)
    test_loader_50, _ = get_dataloader(test_dataset_50, 'testing', args, neighborhood_limits)
    test_loader_70, _ = get_dataloader(test_dataset_70, 'testing', args, neighborhood_limits)
    test_loader_95, _ = get_dataloader(test_dataset_95, 'testing', args, neighborhood_limits)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    logger = SummaryWriter(log_dir=os.path.join(args.tensorboard_path, time.strftime("%m%d-%H%M")))

    best_valid = best_test_30 = best_test_40 = best_test_50 = best_test_70 = best_test_95 = best_30 = best_40 = best_50 = best_70 = best_95 = 0.0
    best_95_history = 0
    best_epoch = 0

    for epoch in range(args.num_epochs):
        train(epoch, train_loader, loss_fn)
        lr_scheduler.step()
        valid_fmax = test(valid_loader)
        test_30 = test(test_loader_30)
        test_40 = test(test_loader_40)
        test_50 = test(test_loader_50)
        test_70 = test(test_loader_70)
        test_95 = test(test_loader_95)

        logger.add_scalar("valid_fmax", valid_fmax, epoch+1)
        logger.add_scalar("test_30", test_30, epoch+1)
        logger.add_scalar("test_40", test_40, epoch + 1)
        logger.add_scalar("test_50", test_50, epoch + 1)
        logger.add_scalar("test_70", test_70, epoch + 1)
        logger.add_scalar("test_95", test_95, epoch + 1)

        print(f'Epoch: {epoch+1:03d}, Validation: {valid_fmax:.4f}, Test: {test_30:.4f}\t{test_40:.4f}\t{test_50:.4f}\t{test_70:.4f}\t{test_95:.4f}\t{best_95_history:.4f}')

        if valid_fmax >= best_valid:
            best_valid = valid_fmax
            best_30 = test_30
            best_40 = test_40
            best_50 = test_50
            best_70 = test_70
            best_95 = test_95
            best_epoch = epoch
            checkpoint = model.state_dict()
            if args.ckpt_path:
                torch.save(checkpoint, osp.join(args.ckpt_path.replace('best.pkl', 'best_{}.pkl'.format(epoch))))
            best_95_history = max(test_95, best_95_history)

        best_test_30 = max(test_30, best_test_30)
        best_test_40 = max(test_40, best_test_40)
        best_test_50 = max(test_50, best_test_50)
        best_test_70 = max(test_70, best_test_70)
        best_test_95 = max(test_95, best_test_95)

    print(f'Best: {best_epoch + 1:03d}, Validation: {best_valid:.4f}, Test: {best_test_30:.4f}\t{best_test_40:.4f}\t{best_test_50:.4f}\t{best_test_70:.4f}\t{best_test_95:.4f}, Valided Test: {best_30:.4f}\t{best_40:.4f}\t{best_50:.4f}\t{best_70:.4f}\t{best_95:.4f}')

