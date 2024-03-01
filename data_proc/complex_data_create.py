import cv2
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
import numpy as np
import argparse
import pickle
import torch
from scipy.spatial.distance import cdist
from tqdm import tqdm

imgN = 3
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

def data_create(img1_path, img2_path,img1_path_i, img2_path_i, img1_path_q, img2_path_q, label_path, h, w):

    data_total = []
    patch_size = 9
    step = 1
    boundary = 16
    sp_label = []
    sp_coord = []
    sp_data = []
    total_num = 0
    pos_num = 0
    neg_num = 0
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    phase1, _ = complex_phase_mag(img1_path_i, img1_path_q, h, w)
    phase2, _ = complex_phase_mag(img2_path_i, img2_path_q, h, w)



    img=cv2.absdiff(img1, img2)
    # cv2.imwrite("img1.png", img)
    labelimg = cv2.imread(label_path)
    labelimg = cv2.cvtColor(labelimg, cv2.COLOR_BGR2GRAY)
    ret, labelimg = cv2.threshold(labelimg, 50, 255, cv2.THRESH_BINARY)
    h, w = labelimg.shape[:2]
    img = img_as_float(img)
    superpixels = slic(img, n_segments=(h*w/100), compactness=20, channel_axis=-1)

    for i in tqdm(range(int(boundary/2), h-int(boundary/2), step), desc='Data_Creating', unit='i'):
        for j in range(int(boundary/2), w-int(boundary/2), step):

            if (labelimg[i][j] == 0 or labelimg[i][j] == 255):
                data = []
                total_num += 1
                if (labelimg[i][j] == 255):
                    pos_num += 1
                    sp_label.append(1)
                    # 构建空时地理数据-正样本
                    img_patch1 = img1[i - int(patch_size / 2):i + int(patch_size / 2) + 1,
                            j - int(patch_size / 2):j + int(patch_size / 2) + 1]
                    img_patch2 = img2[i - int(patch_size / 2):i + int(patch_size / 2) + 1,
                            j - int(patch_size / 2):j + int(patch_size / 2) + 1]
                    img_patch1q = phase1[i - int(patch_size / 2):i + int(patch_size / 2) + 1,
                                 j - int(patch_size / 2):j + int(patch_size / 2) + 1]
                    img_patch2q = phase2[i - int(patch_size / 2):i + int(patch_size / 2) + 1,
                                 j - int(patch_size / 2):j + int(patch_size / 2) + 1]
                    for ii in range(patch_size):
                        for jj in range(patch_size):
                            feature1 = torch.tensor([ii, jj ,1, img_patch1[ii][jj][0]]).float()
                            feature2 = torch.tensor([ii, jj, 2, img_patch2[ii][jj][0]]).float()
                            feature1q = torch.tensor([ii, jj, 1, img_patch1q[ii][jj]]).float()
                            feature2q = torch.tensor([ii, jj, 2, img_patch2q[ii][jj]]).float()
                            feature1 = torch.unsqueeze(feature1, dim=0)
                            feature2 = torch.unsqueeze(feature2, dim=0)
                            feature1q = torch.unsqueeze(feature1q, dim=0)
                            feature2q = torch.unsqueeze(feature2q, dim=0)
                            data.append(feature1)
                            data.append(feature2)
                            data.append(feature1q)
                            data.append(feature2q)
                    data = torch.cat(data, dim=0)
                    data = torch.unsqueeze(data, dim=0)
                    data_total.append(data)
                    #构建空时语义图数据-正样本
                    list_neighbor = []
                    seg = superpixels[i][j]
                    for ii in range(i - int(boundary / 2), i + int(boundary / 2)):
                        for jj in range(j - int(boundary / 2), j + int(boundary / 2)):
                            if (superpixels[ii][jj] == seg):
                                list_neighbor.append([ii, jj])
                    list_neighbor = np.array(list(list_neighbor))
                    distances = cdist(np.array([[i, j]]), list_neighbor)
                    closest_indices = np.argsort(distances)[0][:15]
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

                else:
                    if(total_num % 15 == 0):
                        neg_num += 1
                        sp_label.append(0)
                        # 构建空时地理数据-负样本
                        img_patch1 = img1[i - int(patch_size / 2):i + int(patch_size / 2) + 1,
                                     j - int(patch_size / 2):j + int(patch_size / 2) + 1]
                        img_patch2 = img2[i - int(patch_size / 2):i + int(patch_size / 2) + 1,
                                     j - int(patch_size / 2):j + int(patch_size / 2) + 1]
                        img_patch1q = phase1[i - int(patch_size / 2):i + int(patch_size / 2) + 1,
                                      j - int(patch_size / 2):j + int(patch_size / 2) + 1]
                        img_patch2q = phase2[i - int(patch_size / 2):i + int(patch_size / 2) + 1,
                                      j - int(patch_size / 2):j + int(patch_size / 2) + 1]
                        for ii in range(patch_size):
                            for jj in range(patch_size):
                                feature1 = torch.tensor([ii, jj, 1, img_patch1[ii][jj][0]]).float()
                                feature2 = torch.tensor([ii, jj, 2, img_patch2[ii][jj][0]]).float()
                                feature1q = torch.tensor([ii, jj, 3, img_patch1q[ii][jj]]).float()
                                feature2q = torch.tensor([ii, jj, 4, img_patch2q[ii][jj]]).float()
                                feature1 = torch.unsqueeze(feature1, dim=0)
                                feature2 = torch.unsqueeze(feature2, dim=0)
                                feature1q = torch.unsqueeze(feature1q, dim=0)
                                feature2q = torch.unsqueeze(feature2q, dim=0)
                                data.append(feature1)
                                data.append(feature2)
                                data.append(feature1q)
                                data.append(feature2q)
                        data = torch.cat(data, dim=0)
                        data = torch.unsqueeze(data, dim=0)
                        data_total.append(data)
                        # 构建空时语义图数据 - 负样本
                        list_neighbor = []
                        seg = superpixels[i][j]
                        for ii in range(i - int(boundary / 2), i + int(boundary / 2)):
                            for jj in range(j - int(boundary / 2), j + int(boundary / 2)):
                                if (superpixels[ii][jj] == seg):
                                    list_neighbor.append([ii, jj])
                        list_neighbor = np.array(list(list_neighbor))
                        distances = cdist(np.array([[i, j]]), list_neighbor)
                        closest_indices = np.argsort(distances)[0][:15]
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
    print("pos: {:.4f}".format(pos_num))
    print("neg: {:.4f}".format(neg_num))
    with open('complex_data12_train.pkl', 'wb') as f:
        pickle.dump((sp_label, data_total, sp_data), f, protocol=2)
    # pickle.dump((data_total), f, protocol=2)

