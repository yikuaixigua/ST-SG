import torch
from tqdm import tqdm
from utils.utils import get_optim_lr, AverageMeter, sim_matrix2, compute_diag_sum,plot_samples
import dgl
# from .aug1 import aug_double, collate_batched_graph, sim_matrix2, compute_diag_sum
import torch.nn as nn
from .enhance_data import mask_data, gauss_noise
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# from tensorboardX import SummaryWriter
from .provider import random_point_dropout, random_scale_point_cloud, shift_point_cloud
from train_utils.aug1 import aug_double, collate_batched_graph, sim_matrix2, compute_diag_sum
def train_epoch(excelnum1, worksheet, args, model, loader, epoch, device, optimizer, aug_type='nn'):
    model.train()
    # 记录损失
    total_loss = 0
    rec_loss = AverageMeter()
    # 进度条显示
    loader = tqdm(loader, desc="train [{}/{}]".format(epoch, args.epochs), ncols=100)
    # for iters, (points) in enumerate(loader):
    for iters, (points, labels, batch_graphs, batch_snorm_n, batch_snorm_e) in enumerate(loader):
    # for iters, (points, targets) in enumerate(loader):
        # 清空GPU缓存
        torch.cuda.empty_cache()
        labels = labels.to(device)
        # 数据加载GPU
        points0 = points.to(device)
        batch_graphs.ndata['feat'] = torch.tensor(batch_graphs.ndata['feat'].detach().numpy().T[0])
        batch_graphs = batch_graphs.to(device)
        bg = batch_graphs.to(device)
        bf = batch_graphs.ndata['feat'].to(device)
        be = batch_graphs.edata['feat'].to(device)

    # 时空地理数据增强
        points = points.data.numpy()
        points = random_point_dropout(points)
        points[:, :, 0:3] = random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = shift_point_cloud(points[:, :, 0:3])
        points = torch.Tensor(points)
        points = points.to(device)
        # points0 = points
        points_nosie1 = gauss_noise(points)
        points_nosie2 = gauss_noise(points)
        points_aug1 = mask_data(points_nosie1)
        points_aug2 = mask_data(points_nosie2)

        # 时空语义数据增强
        aug_batch_graphs = dgl.unbatch(batch_graphs)
        aug_list1, aug_list2 = aug_double(aug_batch_graphs, aug_type)
        batch_graphs, batch_snorm_n, batch_snorm_e = collate_batched_graph(aug_list1)
        aug_batch_graphs, aug_batch_snorm_n, aug_batch_snorm_e = collate_batched_graph(aug_list2)

        aug_batch_x = aug_batch_graphs.ndata['feat'].to(device)  # num x feat
        aug_batch_e = aug_batch_graphs.edata['feat'].to(device)
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
        output1 = model(points_aug1, batch_graphs, batch_x, aug_batch_e,
                                   batch_lap_pos_enc, batch_wl_pos_enc)
        output2 = model(points_aug2, aug_batch_graphs, aug_batch_x, batch_e,
                                   batch_lap_pos_enc, batch_wl_pos_enc)
        # output = model(points0, bg, bf, be, batch_lap_pos_enc, batch_wl_pos_enc)
        # loss1 = model.loss(output1, labels)
        # loss2 = model.loss(output2, labels)
        # loss0 = model.loss(output, labels)
        # acc = model.accuracy_MNIST_CIFAR(output, labels) / labels.size(0)
        # nb_data = labels.size(0)
        # outc0 = model(points, bg, bf, be, batch_lap_pos_enc,batch_wl_pos_enc)
        # pca = PCA(n_components=2)
        # reduced_data = pca.fit_transform(outc0.cpu().detach().numpy())
        # fig = plt.figure(figsize=(10, 8))
        # plt.scatter(reduced_data[labels.cpu().detach().numpy().flatten() == 0][:, 0],
        #             reduced_data[labels.cpu().detach().numpy().flatten() == 0][:, 1], label='Cluster 1', alpha=0.5)
        # plt.scatter(reduced_data[labels.cpu().detach().numpy().flatten() == 1][:, 0],
        #             reduced_data[labels.cpu().detach().numpy().flatten() == 1][:, 1], label='Cluster 2', alpha=0.5)
        # # 添加图例
        # plt.legend()
        # # 添加坐标轴标签和标题
        # plt.xlabel('Principal Component 1')
        # plt.ylabel('Principal Component 2')
        # plt.title('2D PCA Projection of the Clustering')
        # plt.show()
        # plot_samples(encoder_(samples).clone().detach(), labels)
        sim_matrix_tmp2 = sim_matrix2(output1, output2, temp=0.0005)
        row_softmax = nn.LogSoftmax(dim=1)
        row_softmax_matrix = -row_softmax(sim_matrix_tmp2)

        colomn_softmax = nn.LogSoftmax(dim=0)
        colomn_softmax_matrix = -colomn_softmax(sim_matrix_tmp2)

        row_diag_sum = compute_diag_sum(row_softmax_matrix)
        colomn_diag_sum = compute_diag_sum(colomn_softmax_matrix)
        contrastive_loss = (row_diag_sum + colomn_diag_sum) / (2 * len(row_softmax_matrix))
        # loss = contrastive_loss * 0.01 + loss0
        # 反向传播
        contrastive_loss.backward()
        optimizer.step()

        # 记录损失
        total_loss += (contrastive_loss.item())
        total_loss0 = total_loss / (iters + 1)
        notes = {"lr": get_optim_lr(optimizer), "loss": total_loss0}
        loader.set_postfix(notes)
        worksheet.write(excelnum1, 1, total_loss0)
        excelnum1 += 1


    return total_loss0, model, optimizer, excelnum1

