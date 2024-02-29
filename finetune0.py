import torch
import argparse
from utils.utils import gpu_setup, load_yaml, load_state_dict, show_yaml, build_save_dir, TensorboardWriter, save_state_dict
from models.finetune_model import create_model
import torch.optim as optim
import os
from data_proc.st_dataset import ST_DatasetLoad
from torch.utils.data import DataLoader, random_split
import xlwt
from finetune_utils.fintune_epoch import finetune_epoch, evaluate_network
from data_proc.data_create_unchanged import data_create
from finetune_utils.predict_unchange import predict_changemap
from data_proc.st_dataset import DGLFormDataset,ST_PixDGL
def get_args():
    """ 执行参数 """
    parser = argparse.ArgumentParser(description='Spatial-Temporal Semantic and Geographic Correlation')

    parser.add_argument('--cfg_path', default='configs.yaml', type=str, help="配置文件路径")


    return parser.parse_args()

def main(args):
    # 加载配置文件
    cfgs = load_yaml(args.cfg_path)

    # 加载数据
    data_create(cfgs['Data']['image1'], cfgs['Data']['image2'], cfgs['Data']['label'])

    # 加载GPU
    device = gpu_setup(cfgs['GPU']['use'], cfgs['GPU']['id'])
    print('cuda available with GPU:', torch.cuda.get_device_name(0))

    # 加载模型
    model = create_model(cfgs, device)
    model.to(device)
    checkpoint = torch.load(cfgs['Train']['pretrained_model'])
    checkpoint_model = checkpoint['module']
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in checkpoint_model.items() if k in model_dict.keys()}
    model.load_state_dict(state_dict, strict=False)
    print('Success load pre-trained model!')
    # 终端打印配置文件参数
    print('参数配置：')
    show_yaml(trace=print, args=cfgs)
    # 创建文件夹, 保存训练的权重
    if cfgs['Train']['finetune']:
        save_dir = os.path.dirname(cfgs['Train']['finetune'])
    else:
        # 保存到 checkpoints 文件夹下
        save_dir = build_save_dir("finetune_checkpoints")

    # 开启tensorbard记录训练损失
    tensor_rec = TensorboardWriter(save_dir)

    # 设置优化器optimizer

    optimizer = optim.Adam(model.parameters(), lr=cfgs['Train']['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfgs['Train']['T_0'],
                                                                     T_mult=cfgs['Train']['T_mult'])

    # 读取数据集
    dataset = ST_DatasetLoad('data.pkl', train_ratio=cfgs['Train']['finetune_train_ratio'], shuffle=True)
    trainset, valset, testset = dataset.train, dataset.val, dataset.test

    train_loader = DataLoader(trainset, batch_size=cfgs['Train']['batchsize'], shuffle=True, drop_last=False,
                              collate_fn=dataset.collate)
    # val_loader = DataLoader(valset, batch_size=cfgs['Train']['batchsize'], shuffle=True, drop_last=False,
    #                           collate_fn=dataset.collate)
    # 加载训练参数
    epochs = cfgs['Train']['epochs']
    args.epochs = epochs
    best_loss = 1e6


    for epoch in range(epochs):
        print('*' * 10, ' epoch [{}/{}] '.format(epoch, epochs), '*' * 10)

        # 训练
        finetune_train_loss, finetune_train_acc, model, optimizer= finetune_epoch(args=args, model=model,
                                                              loader=train_loader, epoch=epoch, device=device, optimizer=optimizer
                                                              )
        # 验证
        # finetune_test_loss, finetune_test_acc, excelnum2 = evaluate_network(excelnum2, worksheet,
        #                                                                     args=args, model=model, loader=val_loader,
        #                                                                     epoch=epoch, device=device)

        scheduler.step()

        # 保存train和valid的loss
        save_state_dict(
            path=os.path.join(save_dir, '{}.pkl'.format("epoch_" + str(epoch))),
            model=model,
            cfgs=args,
            epoch=epoch,
            optim=optimizer,
            min_loss=finetune_train_loss)
        if finetune_train_loss < best_loss:
            best_loss = finetune_train_loss
            save_state_dict(
                path=os.path.join(save_dir, 'best_finetune.pkl'),
                model=model,
                cfgs=args,
                epoch=epoch,
                optim=optimizer,
                min_loss=best_loss)
            print("save to: '{}', finetune min loss: {:.8f}".format(
                os.path.join(save_dir, 'best_finetune.pkl'), best_loss))
        else:
            print("finetune min loss: {:8f}".format(best_loss))
        print()

    # 预测测试数据变化检测结果
    predict_changemap(cfgs['Data']['image1'], cfgs['Data']['image2'], cfgs['Data']['label'],device)





if __name__ == '__main__':
    args = get_args()
    main(args)
