import torch
from tqdm import tqdm
from utils.utils import get_optim_lr, AverageMeter, sim_matrix2, compute_diag_sum
import dgl
# from .aug1 import aug_double, collate_batched_graph, sim_matrix2, compute_diag_sum
import torch.nn as nn
# from tensorboardX import SummaryWriter
from .provider import random_point_dropout, random_scale_point_cloud, shift_point_cloud


def finetune_epoch(args, model, loader, epoch, device, optimizer):
    model.train()
    # 记录损失
    total_loss = 0
    rec_loss = AverageMeter()
    # 进度条显示
    loader = tqdm(loader, desc="finetune [{}/{}]".format(epoch, args.epochs), ncols=100)
    for iters, (points, labels, batch_graphs, batch_snorm_n, batch_snorm_e) in enumerate(loader):
        # for iters, (points, targets) in enumerate(loader):
        # 清空GPU缓存
        torch.cuda.empty_cache()

        # 数据加载GPU
        points = points.data.numpy()
        points = random_point_dropout(points)
        points[:, :, 0:3] = random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = shift_point_cloud(points[:, :, 0:3])
        points = torch.Tensor(points)
        points = points.to(device)
        labels = labels.to(device)
        batch_graphs.ndata['feat'] = torch.tensor(batch_graphs.ndata['feat'].detach().numpy().T[0])
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)

        # 清零梯度
        optimizer.zero_grad()

        try:
            batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
            sign_flip[sign_flip >= 0.5] = 1.0;
            sign_flip[sign_flip < 0.5] = -1.0
            batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        except:
            batch_lap_pos_enc = None

        try:
            batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
        except:
            batch_wl_pos_enc = None

        # 前向传播
        output = model(points, batch_graphs, batch_x, batch_lap_pos_enc, batch_wl_pos_enc)
        loss = model.loss(output, labels)
        acc = model.accuracy_MNIST_CIFAR(output, labels)
        nb_data = labels.size(0)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 记录损失
        total_loss += loss.item()
        total_loss0 = total_loss / (iters + 1)
        acc /= nb_data
        notes = {"lr": get_optim_lr(optimizer), "loss": total_loss0, "acc": acc, }
        loader.set_postfix(notes)


    return total_loss0, acc, model, optimizer


def evaluate_network(args, model, loader, epoch, device):
    model.train()
    # 记录损失
    total_loss = 0
    rec_loss = AverageMeter()
    # 进度条显示
    loader = tqdm(loader, desc="val [{}/{}]".format(epoch, args.epochs), ncols=100)
    for iters, (points, labels, batch_graphs, batch_snorm_n, batch_snorm_e) in enumerate(loader):
        # for iters, (points, targets) in enumerate(loader):
        # 清空GPU缓存
        torch.cuda.empty_cache()

        # 数据加载GPU
        points = points.data.numpy()
        # points = random_point_dropout(points)
        # points[:, :, 0:3] = random_scale_point_cloud(points[:, :, 0:3])
        # points[:, :, 0:3] = shift_point_cloud(points[:, :, 0:3])
        points = torch.Tensor(points)
        points = points.to(device)
        labels = labels.to(device)
        batch_graphs.ndata['feat'] = torch.tensor(batch_graphs.ndata['feat'].detach().numpy().T[0])
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)

        try:
            batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
            sign_flip[sign_flip >= 0.5] = 1.0;
            sign_flip[sign_flip < 0.5] = -1.0
            batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        except:
            batch_lap_pos_enc = None

        try:
            batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
        except:
            batch_wl_pos_enc = None

        # 前向传播
        output = model(points, batch_graphs, batch_x, batch_lap_pos_enc, batch_wl_pos_enc)
        loss = model.loss(output, labels)
        acc = model.accuracy_MNIST_CIFAR(output, labels)
        nb_data = labels.size(0)


        # 记录损失
        total_loss += loss.item()
        total_loss0 = total_loss / (iters + 1)
        acc /= nb_data
        notes = {"valloss": total_loss0,"valacc": acc,}
        loader.set_postfix(notes)


    return total_loss0, acc

