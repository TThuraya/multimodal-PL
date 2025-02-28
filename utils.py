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
#from torch.cuda.amp import autocast

from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss

def get_logger(snapshot_path):
    """
    Creates a logger instance for tracking training progress
    Args:
        snapshot_path: path where log files should be saved
    Returns:
        logger instance
    """
    logger = SummaryWriter(snapshot_path)
    return logger

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, lr, num_stemps, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(lr, i_iter, num_stemps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def extant_file(x):
    """
    'Type' for argparse - checks that file exists but does not open.
    """
    if not os.path.exists(x):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x

def all_reduce_tensor(tensor, world_size=1, norm=True):
    # Since we're CPU-only, just return the mean
    return torch.mean(tensor)

def mask_aug(mask, aug_times=2):
    """
    Augments segmentation masks by duplicating them for multi-augmentation training.
    Used in conjunction with image augmentation to create multiple training pairs
    from a single input sample.
    
    Args:
        mask: numpy array of shape [B, 1, D, H, W]
            B: batch size
            1: single channel for segmentation labels
            D, H, W: depth, height, width of the 3D volume
        aug_times: int, number of copies to create for each mask (default: 2)
            When aug_times=2, each mask will have two copies for training
    
    Returns:
        aug_mask: numpy array of shape [B*aug_times, 1, D, H, W]
            Duplicated masks arranged sequentially
            For batch_size=1, aug_times=2: [mask1, mask1]
            For batch_size=2, aug_times=2: [mask1, mask1, mask2, mask2]
    
    Example:
        Input mask shape: [2, 1, 64, 192, 192] (batch_size=2)
        aug_times = 2
        Output shape: [4, 1, 64, 192, 192]
        Where each original mask is duplicated once
    """
    if aug_times <= 1:
        return mask
        
    orig_shape = mask.shape
    aug_shape = (orig_shape[0] * aug_times,) + orig_shape[1:]
    aug_mask = np.zeros(aug_shape, dtype=mask.dtype)
    
    # Duplicate each sample in the batch aug_times times
    for i in range(orig_shape[0]):  # for each item in batch
        for j in range(aug_times):
            aug_mask[i*aug_times + j] = mask[i]
            
    return aug_mask

def seedfix(seed):
    """
    Sets random seeds for reproducible experiments.
    Called at the start of training and evaluation.
    
    Args:
        seed: int
            Random seed value, typically passed via command line args
            Default is 0 in training/evaluation scripts
    
    Effects:
        - Sets Python's random module seed
        - Sets NumPy's random number generator seed
        - Sets PyTorch's random number generator seed
        - Ensures deterministic behavior in neural network operations
    
    Example:
        In training:
        >>> args = parser.parse_args()
        >>> seedfix(args.seed)  # Set seeds before any random operations
        >>> model = create_model()  # Model weights will initialize the same way
        >>> train_loader = create_dataloader()  # Data augmentation will be reproducible
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Make PyTorch operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False