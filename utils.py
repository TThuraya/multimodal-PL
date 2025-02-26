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


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, lr, num_stemps, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(lr, i_iter, num_stemps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr