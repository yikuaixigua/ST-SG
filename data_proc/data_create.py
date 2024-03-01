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


def data_create(img1_path, img2_path, label_path, segsize, patch_size=9, k=15):

    data_total = []
    # patch_size = 9
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
    img = cv2.absdiff(img1, img2)
    # cv2.imwrite("img1.png", img)
    labelimg = cv2.imread(label_path)
    labelimg = cv2.cvtColor(labelimg, cv2.COLOR_BGR2GRAY)
    ret, labelimg = cv2.threshold(labelimg, 50, 255, cv2.THRESH_BINARY)
    h, w = labelimg.shape[:2]
    img = img_as_float(img)
    superpixels = slic(img, n_segments=(h*w/segsize), compactness=20, channel_axis=-1)

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
                    for ii in range(patch_size):
                        for jj in range(patch_size):
                            feature1 = torch.tensor([ii, jj ,1, img_patch1[ii][jj][0]]).float()
                            feature2 = torch.tensor([ii, jj, 2, img_patch2[ii][jj][0]]).float()
                            feature1 = torch.unsqueeze(feature1, dim=0)
                            feature2 = torch.unsqueeze(feature2, dim=0)
                            data.append(feature1)
                            data.append(feature2)
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

                else:
                    if(total_num % 7 == 0):
                        neg_num += 1
                        sp_label.append(0)
                        # 构建空时地理数据-负样本
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
                        # 构建空时语义图数据 - 负样本
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
    print("pos: {:.4f}".format(pos_num))
    print("neg: {:.4f}".format(neg_num))
    with open('data'+str(segsize) + str(patch_size) + str(k) + '.pkl', 'wb') as f:
        pickle.dump((sp_label, data_total, sp_data), f, protocol=2)
    # pickle.dump((data_total), f, protocol=2)

