import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from models.finetune_model import create_model
from scipy.spatial.distance import cdist
import argparse, json
from sklearn.decomposition import PCA
import pickle
from utils.utils import gpu_setup, load_yaml, load_state_dict
from data_proc.st_dataset import ST_DatasetLoad
import warnings
from skimage.util import img_as_float
from skimage.segmentation import slic, mark_boundaries
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import torch
from oa import evaluation
imgN = 3
from thop import clever_format, profile

warnings.filterwarnings("ignore")
def get_args():
    """ 执行参数 """
    parser = argparse.ArgumentParser(description='Optimizing 3d Time-Series Data (Fast Algorithm to FDTD)')

    parser.add_argument('--cfg_path', default='configs.yaml', type=str, help="配置文件路径")

    parser.add_argument('--device', default='1', nargs='+', help="训练GPU id")

    parser.add_argument('--local_rank', default=-1, type=int, help='多GPU训练固定参数')

    print('cuda available with GPU:', torch.cuda.get_device_name(0))
    return parser.parse_args()

def Fill_border(image, border_size):
    # 获取原始图像的尺寸
    height, width = image.shape[:2]
    border_size = int(border_size)

    # 创建一个新的画布，尺寸为原始图像尺寸加上边框大小的两倍
    new_height = height + 2 * border_size
    new_width = width + 2 * border_size
    bordered_image = np.zeros((int(new_height), int(new_width), 3), dtype=np.uint8)

    # 将原始图像放置在新画布中心位置
    bordered_image[border_size:height+border_size, border_size:width+border_size] = image

    # 在四周添加黑色边框
    bordered_image[:border_size, :] = 0  # 上边框
    bordered_image[height+border_size:, :] = 0  # 下边框
    bordered_image[:, :border_size] = 0  # 左边框
    bordered_image[:, width+border_size:] = 0  # 右边框

    return bordered_image


def change_2(matrix):
    num_elements = matrix.size
    num_to_zero = int(num_elements * 0.13)

    # 找到要置0的元素的索引
    flatten_matrix = matrix.flatten()
    sorted_indices = np.argsort(flatten_matrix)
    indices_to_zero = np.concatenate((sorted_indices[:0], sorted_indices[-num_to_zero:]))

    # 将要置0的元素置为0
    flatten_matrix[indices_to_zero] = 0.1

    # 重新构造矩阵
    matrix_zeroed = flatten_matrix.reshape(matrix.shape)

    return matrix_zeroed
def complex_convolution(image):
    # 定义卷积核
    window_size = 17
    w1 = int(window_size / 2)
    w2 = int(window_size / 2) + 1
    # kernel = np.ones((window_size, window_size), dtype=np.complex128)
    # kernel[1, 1] = 0 + 0j
    kernel = np.ones((window_size, window_size), dtype=np.int16)
    kernel[1, 1] = 0

    # 获取图像尺寸
    height, width = image.shape

    # 创建一个与原图像相同尺寸的输出图像
    output = np.zeros_like(image, dtype=np.complex128)

    # 进行卷积操作
    for i in range(w1, height - w1):
        for j in range(w1, width - w1):
            # 提取 3x3 区域
            region = image[i-w1:i+w2, j-w1:j+w2]
            # 对区域与卷积核进行乘法计算并求和
            convolved_value = np.sum(region * kernel)
            # 将计算结果赋值给输出图像的对应位置
            output[i, j] = convolved_value

    return output
def complex_phase_mag(img1_path_i, img1_path_q, h, w):
    data_i = np.fromfile(img1_path_i, dtype=np.int16)
    data_q = np.fromfile(img1_path_q, dtype=np.int16)
    rows = h
    cols = w
    scale = 1
    # 将一维数据转换为矩阵形式
    matrix_i = np.reshape(data_i * scale, (rows, cols))
    matrix_q = np.reshape(data_q * scale, (rows, cols))
    matrix_i = change_2(matrix_i)
    matrix_q = change_2(matrix_q)
    matrix_z = matrix_i + 1j * matrix_q
    # z = complex(matrix_i, matrix_q)

    phase = np.angle(matrix_z)  # 相位图
    # phase = change_2(phase)
    # 将相位值转换为角度（弧度转换为度数）
    phase_degrees = np.degrees(phase)
    phase_degrees = change_2(phase_degrees)
    cos_img = np.cos(phase_degrees)
    sin_img = np.sin(phase_degrees)
    complex_matrix = np.vectorize(complex)(cos_img, sin_img)

    phase_b = np.angle(complex_matrix)  # 相位图
    # phase_b = change_2(phase_b)
    # 将相位值转换为角度（弧度转换为度数）
    phase_degrees_b = np.degrees(phase_b)

    result_complex = complex_convolution(complex_matrix)

    phase_a = np.angle(result_complex)
    # phase_a = change_2(phase_a)
    # 将相位值转换为角度（弧度转换为度数）
    phase_degrees_a = np.degrees(phase_a)

    phase_degrees_diff = phase_degrees_b - phase_degrees_a
    min_value = phase_degrees_diff.min()
    max_value = phase_degrees_diff.max()

    # 归一化数组到0-1
    phase_degrees_diff = (phase_degrees_diff - min_value) / (max_value - min_value)

    magnitude = np.abs(matrix_z)
    percentile_98th = np.percentile(magnitude, 99.5)
    clipped_array = np.where(magnitude > percentile_98th, percentile_98th, magnitude)
    min_value = clipped_array.min()
    max_value = clipped_array.max()
    normalized_array = (clipped_array - min_value) / (max_value - min_value) * 255
    return phase_degrees_diff, normalized_array



def predict_changemap(img1_path, img2_path, label_path,segsize, patch_size, k , path,  device):
    start = time.time() 
    # 加载配置文件
    args = get_args()

    cfgs = load_yaml(args.cfg_path)

    # 加载模型

    print('#########Success load trained model!#########')
    model = create_model(cfgs, device)
    model, *unuse = load_state_dict(path, model)
    model.eval().to(device)

    # # 输入双时相图像数据，计算差分图
    boundary = 16
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img = cv2.absdiff(img1, img2)
    # img = cv2.imread("diff14.bmp")
    label = cv2.imread(label_path)
    img1 = Fill_border(img1, boundary/2)
    img2 = Fill_border(img2, boundary/2)
    img = Fill_border(img, boundary/2)
    label = Fill_border(label, boundary/2)
    # patch_size = 9
    step = 1
    h, w = img.shape[:2]
    img = img_as_float(img)
    #  注释
    superpixels = slic(img, n_segments=int((h*w/segsize)), compactness=20, channel_axis=-1)

    data_total = []
    sp_label = []
    sp_data = []
    for i in tqdm(range(int(boundary/2), h-int(boundary/2), step), desc='Test_Data_Creating', unit='i'):
        for j in range(int(boundary / 2), w - int(boundary / 2), step):
            data = []
            sp_label.append(label[i][j][0])
            # 构建空时地理数据
            img_patch1 = img1[i - int(patch_size / 2):i + int(patch_size / 2) + 1,
                         j - int(patch_size / 2):j + int(patch_size / 2) + 1]
            img_patch2 = img2[i - int(patch_size / 2):i + int(patch_size / 2) + 1,
                         j - int(patch_size / 2):j + int(patch_size / 2) + 1]
            for ii in range(patch_size):
                for jj in range(patch_size):
                    feature1 = torch.tensor([ii, jj, 1, img_patch1[ii][jj][0]]).float()
                    feature2 = torch.tensor([ii, jj, 2, img_patch2[ii][jj][0]]).float()
                    feature1 = torch.unsqueeze(feature1, dim=0)
                    feature2 = torch.unsqueeze(feature2, dim=0)
                    data.append(feature1)
                    data.append(feature2)
            data = torch.cat(data, dim=0)
            data = torch.unsqueeze(data, dim=0)
            data_total.append(data)
            # 构建空时语义图数据
            list_neighbor = []
            seg = superpixels[i][j]
            for ii in range(i - int(boundary / 2), i + int(boundary / 2)):
                for jj in range(j - int(boundary / 2), j + int(boundary / 2)):
                    if (superpixels[ii][jj] == seg):
                        list_neighbor.append([ii, jj])
            list_neighbor = np.array(list(list_neighbor))
            distances = cdist(np.array([[i, j]]), list_neighbor)
            closest_indices = np.argsort(distances)[0][:k]
            closest_points = list_neighbor[closest_indices]
            sp_coord0 = []
            sp_intensity = []
            for n in range(len(closest_points)):
                coord = closest_points[n]
                sp_coord0.append(coord)
                intensity = img[coord[0]][coord[1]][0]
                sp_intensity.append(intensity)
            sp_intensity = np.array(sp_intensity, np.float32)
            sp_intensity = torch.from_numpy(sp_intensity)
            sp_coord = np.array(sp_coord0, np.float32)
            sp_data.append([sp_intensity, sp_coord])

    sp_label = np.array(sp_label, np.int32)
    data_total = torch.cat(data_total, dim=0)
    with open('dataset_predict' + str(segsize) + '.pkl', 'wb') as f:
        pickle.dump((sp_label, data_total, sp_data), f, protocol=2)
    print('#########Success save predict dataset!#########')
     # 注释
    # 加载数据
    dataset = ST_DatasetLoad('dataset_predict' +  str(segsize) + '.pkl', train_ratio = 1, shuffle=False)
    trainset, valset, testset = dataset.train, dataset.val, dataset.test
    print('#########Success load predict dataset!#########')
    train_loader = DataLoader(trainset, batch_size=100, shuffle=False, drop_last=False,
                              collate_fn=dataset.collate)
    target_array = np.zeros((h, w))
    label_total = [ ]

    for iters, (points, labels, batch_graphs, batch_snorm_n, batch_snorm_e) in tqdm(enumerate(train_loader), desc='Test_Data_Predicting', unit='iters'):
        # 清空GPU缓存
        torch.cuda.empty_cache()

        # 数据加载GPU
        points = points.data.numpy()
        points = torch.Tensor(points)
        points = points.to(device)
        batch_graphs.ndata['feat'] = torch.tensor(batch_graphs.ndata['feat'].detach().numpy().T[0])
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        labels = labels.to(device)
        try:
            batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        except:
            batch_lap_pos_enc = None

        try:
            batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
        except:
            batch_wl_pos_enc = None


        output = model(points, batch_graphs, batch_x, batch_lap_pos_enc, batch_wl_pos_enc).detach().argmax(
            dim=1)
        label_total.append(output)

    label_total = torch.cat(label_total, dim=0)
    label_total = label_total.reshape(h - 2 * int(boundary / 2), w - 2 * int(boundary / 2))
    label_array = label_total.cpu().numpy()
    normalized_array = np.interp(label_array, (label_array.min(), label_array.max()), (0, 255))
    image_array = normalized_array.astype(np.uint8)
    cv2.imwrite('changemap0_18_real' +  str(segsize)+ str(patch_size) + str(k) + '.bmp', image_array)
    _, binary_image = cv2.threshold(image_array, 127, 255, cv2.THRESH_BINARY)
    area_threshold = 5
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    cleaned_image = np.zeros_like(binary_image)
    for label in range(1, num_labels):  # 从1开始，0是背景
        area = stats[label, cv2.CC_STAT_AREA]
        if area > area_threshold:
            cleaned_image[labels == label] = 255
    kernel = np.ones((3, 3), np.uint8)
    cleaned_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_OPEN, kernel)
    cv2.imwrite('changemap18_real' + str(segsize) + str(patch_size) + str(k) + '.bmp', cleaned_image)
    end = time.time()
    total_time = end - start
    print('Running time: %s Seconds' % (end - start))
    print('######### Predict Over!#########')
    truth = cv2.imread(r'datasets/label18.bmp')
    f, miou, acc = evaluation(image_array, truth)
    return f, miou, acc, total_time
