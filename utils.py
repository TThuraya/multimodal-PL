import argparse
import os, sys

sys.path.append("..")

import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import datetime
import os.path as osp
#from unet3D_DynConv882 import UNet3D, unet3D_with_eam, get_style_discriminator_linear, unet3D_with_feam, unet3D_with_feam2, get_style_discriminator_output, norm_style_discriminator_output, deep_style_discriminator_output, unet3D_with_deepsup, unet3D_g
#from MOTSDataset import MOTSDataSet, my_collate, AMOSDataSet, AMOSDataSet_newatlas, AMOSDataSet_newatlas_alldata, get_mask_dict, get_modal_dict, AMOSDataSet_newatlas_onlyct, get_mask_dict_ct

import random
import timeit
from tensorboardX import SummaryWriter
##from loss_functions import loss
#from loss_partial import EDiceLoss_partial, EDiceLoss_full, EDiceLoss_full2
#from evaluate_amos import predict_sliding, get_dice, get_dice2

from sklearn import metrics
from math import ceil

#from engine import Engine
#from apex import amp
#from apex.parallel import convert_syncbn_model

import csv
from torch.cuda.amp import autocast

from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, lr, num_stemps, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(lr, i_iter, num_stemps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def masking(B, mask_ratio = 0.25, patch_size = 16, raw_size = [64, 192, 192]):

    token_count = raw_size[0] // patch_size * raw_size[1] // patch_size * raw_size[2] // patch_size
    token_index = [x for x in range(0, token_count)]
    temp_index = token_index.copy()
    masked_list = []

    sample_length = int(token_count * mask_ratio)

    while len(masked_list) < sample_length:
        sample = random.choice(temp_index)
        masked_list.append(sample)
        temp_index.remove(sample)
    
    decoder_embeeding = torch.zeros((B, token_count, patch_size, patch_size, patch_size))
    decoder_embeeding[:,  masked_list, :, :, :] = 1

    decoder_embeeding = decoder_embeeding.reshape((B, 1, raw_size[0] // patch_size, raw_size[1] // patch_size, raw_size[2] // patch_size, patch_size, patch_size, patch_size)).permute(0, 1, 2, 5, 3, 6, 4, 7)
    decoder_embeeding = decoder_embeeding.reshape((B, 1, raw_size[0], raw_size[1], raw_size[2]))

    return decoder_embeeding

def mask_(cpred):

    if np.random.random() < 0.5:
        mask_r = 0.125#0.03125
    else:
        mask_r = 0.0625#0.015625

    if np.random.random() < 0.5:
        patch_size = 16
    else:
        patch_size = 16

    for l in range(cpred.shape[0]):
        if np.random.random() < 0.5:

            mask0 = masking(1, mask_r, patch_size, raw_size = [cpred.shape[2], cpred.shape[3], cpred.shape[4]]).to(cpred.device)
            cpred[l:l+1] = cpred[l:l+1] * (1-mask0)

        if np.random.random() < 0.5:

            mask1 = masking(1, mask_r, patch_size, raw_size = [cpred.shape[2], cpred.shape[3], cpred.shape[4]]).to(cpred.device)
            cpred[l:l+1] = cpred[l:l+1] * (1-mask1) + torch.ones_like(cpred[l:l+1]) * mask1
    return cpred

def mask_aug(cpred, duplicate = 1):
    # cpred : b * c * h * w * d

    if duplicate == 1:
        return mask_(cpred)
    else:
        pred_f = []
        for l in range(duplicate):
            pred_f.append(mask_(cpred))
        return torch.cat(pred_f, dim = 0)


def seedfix(seed):
    if seed == 0:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        return
    torch.manual_seed(seed) # 为CPU设置随机种子
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.	
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True