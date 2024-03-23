import torch
import numpy as np
def gauss_noise(points, p = 0.1):
    noise = torch.randn_like(points)
    exp_noise = torch.exp(p * noise)
    x_noisy = points * exp_noise
    return x_noisy

def mask_data(points):
    total_elements = points.numel()
    num_zeros = int(total_elements * 0.2)
    mask1 = torch.zeros_like(points)
    indices1 = torch.randperm(total_elements)[:num_zeros]
    mask1.view(-1)[indices1] = 1
    mask1 = mask1.cuda()
    points_aug1 = points * mask1
    return points_aug1
