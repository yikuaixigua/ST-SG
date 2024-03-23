import torch
import argparse
from utils.utils import gpu_setup, load_yaml, load_state_dict, show_yaml, build_save_dir, TensorboardWriter, save_state_dict
from models.model import create_model
import torch.optim as optim
import os
from data_proc.st_dataset import ST_DatasetLoad
from torch.utils.data import DataLoader
import xlwt
from train_utils.train_epoch import train_epoch
from torch.nn.parallel import DataParallel
import torch.distributed as dist
from data_proc.st_dataset import DGLFormDataset,ST_PixDGL
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'
dist.init_process_group(backend='nccl',init_method='env://',rank=0,world_size=int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1)

def get_args():
    """ 执行参数 """
    parser = argparse.ArgumentParser(description='spatial-time graph network with cl and transformer')

    parser.add_argument('--cfg_path', default='configs.yaml', type=str, help="配置文件路径")

    parser.add_argument('--device', default='1', nargs='+', help="训练GPU id")

    parser.add_argument('--local_rank', default=0, type=int, help='多GPU训练固定参数')

    print('cuda available with GPU:', torch.cuda.get_device_name(0))

    return parser.parse_args()

def main(args):
    # 加载配置文件
    cfgs = load_yaml(args.cfg_path)

    torch.cuda.set_device(args.local_rank)
    # torch.distributed.init_process_group(
    #     'nccl',
    #     init_method='env://'
    # )
    if isinstance(args.device, (int, str)):
        device_ids = [args.device]
    elif isinstance(args.device, (list, tuple)):
        new_list = [list(map(int, item)) for item in args.device]
        device_ids = [int(x) for sublist in new_list for x in sublist]
    device = torch.device("cuda:{}".format(device_ids[args.local_rank]))
    # 加载模型
    model = create_model(cfgs,device)
    if cfgs['Train']['resume']:
        model, *unused = load_state_dict(cfgs['Train']['resume'], model)
    model.to(device)
    # model = DataParallel(model, device_ids=[0, 1])
    # model.to('cuda')

    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    # 终端打印配置文件参数
    print('参数配置：')
    show_yaml(trace=print, args=cfgs)
    # 创建文件夹, 保存训练的权重
    if cfgs['Train']['resume']:
        save_dir = os.path.dirname(cfgs['Train']['resume'])
    else:
        # 保存到 checkpoints 文件夹下
        save_dir = build_save_dir("checkpoints")
    save_best_dir = os.path.dirname(cfgs['Train']['best_path'])
    # 开启tensorbard记录训练损失
    tensor_rec = TensorboardWriter(save_dir)

    # 设置优化器optimizer

    optimizer = optim.Adam(model.parameters(), lr=cfgs['Train']['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfgs['Train']['T_0'],
                                                                     T_mult=cfgs['Train']['T_mult'])

    # 读取数据集
    dataset = ST_DatasetLoad('data_proc/dataset_train15.pkl', train_ratio=1, shuffle=True)
    trainset, valset, testset = dataset.train, dataset.val, dataset.test
    # train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    train_loader = DataLoader(trainset, batch_size=cfgs['Train']['batchsize'], shuffle=True, drop_last=False,
                              collate_fn=dataset.collate)

    # train_loader = DataLoader(trainset, batch_size=cfgs['Train']['batchsize'], sampler=train_sampler, drop_last=False,
    #                           collate_fn=dataset.collate)
    # # 加载训练参数
    epochs = cfgs['Train']['epochs']
    args.epochs = epochs
    best_loss = 1e6

    # 创建一个excel文件，用于存储loss到excel文件中
    workbook = xlwt.Workbook(encoding='utf-8')
    # 创建一个worksheet
    worksheet = workbook.add_sheet('loss_value')
    excelnum1 = 1
    excelnum2 = 1
    for epoch in range(epochs):
        print('*' * 10, ' epoch [{}/{}] '.format(epoch, epochs), '*' * 10)

        # 训练
        train_loss, model, optimizer, excelnum1 = train_epoch(excelnum1, worksheet, args=args, model=model,
                                                              loader=train_loader, epoch=epoch, device=device, optimizer=optimizer
                                                              , aug_type='nn')

        scheduler.step()

        # tensorboard写入训练损失
        tensor_rec.writer.add_scalar('train loss', train_loss, epoch)

        # 保存train和valid的loss
        workbook.save('loss_train15_00005.xls')
        # save_state_dict(
        #     path=os.path.join(save_dir, '{}.pkl'.format("epoch_" + str(epoch))),
        #     model=model,
        #     cfgs=args,
        #     epoch=epoch,
        #     optim=optimizer,
        #     min_loss=train_loss)
        if train_loss < best_loss:
            best_loss = train_loss
            save_state_dict(
                path=os.path.join(save_best_dir, 'best_pre15_00005.pkl'),
                model=model,
                cfgs=args,
                epoch=epoch,
                optim=optimizer,
                min_loss=best_loss)
            print("save to: '{}', min loss: {:.8f}".format(
                os.path.join(save_best_dir, 'best_15_00005.pkl'), best_loss))
        else:
            print("train min loss: {:8f}".format(best_loss))
        print()




if __name__ == '__main__':
    args = get_args()
    main(args)