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
from loss_functions.loss_partial import EDiceLoss_partial, EDiceLoss_full, EDiceLoss_full2

#from evaluate_amos import predict_sliding, get_dice, get_dice2

from sklearn import metrics
from math import ceil

from engine import Engine
#from apex import amp
#from apex.parallel import convert_syncbn_model

import csv
#from torch.cuda.amp import autocast

from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss

from utils import adjust_learning_rate, mask_aug



def get_loss_refine(output, label, dlist, aug_mask = 1):

    loss = 0.
    cedice = EDiceLoss_full(2)
    if aug_mask <= 1:

        for idx, l in enumerate(dlist):
            # print(l, output.shape)
            #print(output.shape, label.shape)
            loss += cedice(output[idx:idx+1], (label == (l+1)).squeeze(1), uce=False)
    else:
        for kk in range(aug_mask):
            start = kk * len(dlist)
            #print(output.shape, start, sum(dlist), kk, dlist)
            for idx, l in enumerate(dlist):
                loss += cedice(output[start + idx: start + idx+1], (label == (l+1)).squeeze(1), uce=False)
    return loss

def get_loss_mse(output, cm, deep_out, target, mask = None, catlas = None, attns = None, refine_output = None, label_t = None):
        total_loss = 0.0
        
        # Starting here it gets spicy!
        dice_loss = edice(output, target.squeeze(1), soft_max = True, mask = mask)

        aux_loss = 0.0
        weights = [0.03, 0.1, 0.2, 0.3]
        if len(deep_out) != 0:

            for idx, l in enumerate(deep_out):

                aatlas = nn.functional.interpolate(target, l.shape[2:], mode='nearest').float()

                aux_loss +=  edice(l, aatlas.squeeze(1), soft_max = True, mask = mask, uce=False) * weights[idx]
        
        if refine_output is not None:


            refine_output = torch.softmax(refine_output, 1)

            attns.append(output[:, 1:])
            for idx, l in enumerate(attns):

                cedice = EDiceLoss_full2(2)

                c_refine_label = nn.functional.interpolate(refine_output, l.shape[2:]).float()
                l_r = torch.softmax(l, dim = 1)
                
                for gan in range(8):
                    if not label_t[gan]:
                #         aux_loss += 
                        aux_loss += F.mse_loss(l_r[0, gan], c_refine_label[gan, 1]) / 7 * weights[idx]
            
            attns.pop(-1)
        else:
            for l in attns:
                aux_loss += l.mean() * 0

        total_loss += dice_loss + aux_loss

        return total_loss

def get_loss(output, cm, deep_out, target, mask = None, catlas = None, attns = None, refine_output = None, label_t = None, discard = 0.05, confi_ = 0.10, aux_weight = 1, weight_feature = 0.1):
        total_loss = 0.0
        edice = EDiceLoss_partial(output.shape[1])
        num_classes = output.shape[1] - 1

        # Starting here it gets spicy!
        dice_loss = edice(output, target.squeeze(1), soft_max = True, mask = mask)

        aux_loss = 0.0
        weights = [0.125, 0.25, 0.5, 1]

        dices = []
        if len(deep_out) != 0:

            for idx, l in enumerate(deep_out):

                ctarget = nn.functional.interpolate(target, l.shape[2:], mode='nearest').float()

                cdice = edice(l, ctarget.squeeze(1), soft_max = True, mask = mask, uce=False)

                dices.append(cdice.detach().cpu().numpy())

                aux_loss +=  cdice * weights[idx]
        
        if refine_output is not None:


            refine_label = torch.zeros_like(target, device = target.device)

            refine_output_p = torch.softmax(refine_output, 1)

            #confi_mask = confi > 0.3
            confi_mask = torch.logical_or(refine_output_p > (1-confi_), refine_output_p < confi_).float()

            confi_ = 0.10

            refine_output = torch.argmax(refine_output_p, dim = 1).unsqueeze(1)

            supcount = 0

            for l in range(refine_output.shape[0]):
                if label_t[l]:
                    supcount += 1
                else:
                    refine_label[refine_output[l:l+1] == 1] = l+1

            for l in range(refine_output.shape[0]):
                if label_t[l]:
                    refine_label[target == l+1] = l+1
                else:
                    pass
            cedice = EDiceLoss_full2(2)
            attns.append(torch.softmax(output, 1)[:, 1:])
            dices = []
            for idx, l in enumerate(attns):
                
                for gan in range(num_classes):
                    if not label_t[gan]:

                        if idx == 3:
                            cdice = cedice(l[:, gan:gan+1], refine_output_p[gan:gan+1, 1], uce=False, sigmoid = False, mask = confi_mask[gan:gan+1, 1:])
                        else:
                            cdice = cedice(l[:, gan:gan+1], refine_output_p[gan:gan+1, 1], uce=False, mask = confi_mask[gan:gan+1, 1:])

                        dices.append(cdice.detach().cpu().numpy().round(3).item())
                        aux_loss += cdice  / (num_classes - supcount) * weights[idx] * weight_feature

            attns.pop(-1)

            total_loss += dice_loss + aux_loss * aux_weight

            return total_loss, confi_
        else:
            # print("xxk")
            # pass
            return dice_loss, confi_

def get_loss2(output, cm, deep_out, target, mask = None, catlas = None, attns = None, refine_output = None, label_t = None, discard = 0.05, confi_ = 0.10, aux_weight = 1):
        total_loss = 0.0
        edice = EDiceLoss_partial(output.shape[1])
        num_classes = output.shape[1] - 1

        # Starting here it gets spicy!
        dice_loss = edice(output, target.squeeze(1), soft_max = True, mask = mask)

        aux_loss = 0.0
        weights = [0.125, 0.25, 0.5, 1]

        dices = []
        if len(deep_out) != 0:

            for idx, l in enumerate(deep_out):

                ctarget = nn.functional.interpolate(target, l.shape[2:], mode='nearest').float()

                cdice = edice(l, ctarget.squeeze(1), soft_max = True, mask = mask, uce=False)

                dices.append(cdice.detach().cpu().numpy())

                aux_loss +=  cdice * weights[idx]
        #print(dices)
        
        if refine_output is not None:


            refine_label = torch.zeros_like(target, device = target.device)

            refine_output_p = torch.softmax(refine_output, 1)

            #confi = - refine_output.detach() * torch.log(refine_output.detach())
            #confi_mask = confi > 0.3
            confi_mask = torch.logical_or(refine_output_p > (1-confi_), refine_output_p < confi_).float()

            # print(confi_mask.shape, refine_output.shape) # 8 2 patch size
            # (confi_mask.shape, confi_mask.sum(), confi_)
            #if confi_mask.sum() / 37748736 > 1-discard and confi_>0.02:
            #    confi_ = confi_ - 0.001
            #else:
            #    confi_ = confi_ + 0.001
            confi_ = 0.10

            refine_output = torch.argmax(refine_output_p, dim = 1).unsqueeze(1)

            supcount = 0

            for l in range(refine_output.shape[0]):
                if label_t[l]:
                    supcount += 1
                else:
                    refine_label[refine_output[l:l+1] == 1] = l+1

            for l in range(refine_output.shape[0]):
                if label_t[l]:
                    refine_label[target == l+1] = l+1
                else:
                    pass
            cedice = EDiceLoss_full2(2)
            attns.append(output[:, 1:])
            dices = []
            for idx, l in enumerate(attns):
                

                for gan in range(num_classes):
                    if not label_t[gan]:

                        if idx == 5:  # never
                            cdice = cedice(l[:, gan:gan+1], refine_output_p[gan:gan+1, 1], uce=False, sigmoid = False, mask = confi_mask[gan:gan+1, 1:])
                        else:
                            cdice = cedice(l[:, gan:gan+1], refine_output_p[gan:gan+1, 1], uce=False, mask = confi_mask[gan:gan+1, 1:])
                        # cdice = edice()
                        #if idx >= 3:
                        dices.append(cdice.detach().cpu().numpy().round(3).item())
                        aux_loss += cdice  / (num_classes - supcount) * weights[idx] * 0.1
                
            # print(dices)
            attns.pop(-1)

            total_loss += dice_loss + aux_loss * aux_weight

            return total_loss, confi_
        else:
            # print("xxk")
            # pass
            return dice_loss, confi_

def get_loss_multiref(output, cm, deep_out, target, mask = None, catlas = None, attns = None, refine_output = None, label_t = None, discard = 0.05, confi_ = 0.15, aux_weight = 1):
        total_loss = 0.0
        edice = EDiceLoss_partial(output.shape[1])
        num_classes = output.shape[1] - 1

        # Starting here it gets spicy!
        dice_loss = edice(output, target.squeeze(1), soft_max = True, mask = mask)

        aux_loss = 0.0
        weights = [0.125, 0.25, 0.5, 1]

        dices = []
        if len(deep_out) != 0:

            for idx, l in enumerate(deep_out):

                ctarget = nn.functional.interpolate(target, l.shape[2:], mode='nearest').float()

                cdice = edice(l, ctarget.squeeze(1), soft_max = True, mask = mask, uce=False)

                dices.append(cdice.detach().cpu().numpy())

                aux_loss +=  cdice * weights[idx]
        #print(dices)
        
        if refine_output is not None:


            refine_label = torch.zeros_like(target, device = target.device)

            refine_output_p = torch.softmax(refine_output, 1)

            #confi = - refine_output.detach() * torch.log(refine_output.detach())
            #confi_mask = confi > 0.3
            confi_mask = torch.logical_or(refine_output_p > (1-confi_), refine_output_p < confi_).float()

            # print(confi_mask.shape, refine_output.shape) # 8 2 patch size
            # (confi_mask.shape, confi_mask.sum(), confi_)
            #if confi_mask.sum() / 37748736 > 1-discard and confi_>0.02:
            #    confi_ = confi_ - 0.001
            #else:
            #    confi_ = confi_ + 0.001
            confi_ = 0.15

            refine_output = torch.argmax(refine_output_p, dim = 1).unsqueeze(1)

            supcount = 0

            for l in range(refine_output.shape[0]):
                if label_t[l]:
                    supcount += 1
                else:
                    refine_label[refine_output == l+1] = l+1

            for l in range(refine_output.shape[0]):
                if label_t[l]:
                    refine_label[target == l+1] = l+1
                else:
                    pass

            attns.append(output[:, 1:])
            dices = []
            for idx, l in enumerate(attns):
                
                #print(idx)
                #if idx != len(attns) - 1:
                #    continue

                c_refine_label = nn.functional.interpolate(refine_label, l.shape[2:], mode='nearest').float()
                #c_confi_mask = nn.functional.interpolate(confi_mask, l.shape[2:], mode = 'nearest').float()
                cedice = EDiceLoss_full2(2)

                for gan in range(num_classes):
                    if not label_t[gan]:
                #         aux_loss += 
                        # print(refine_output_p.shape, catlas.shape)
                        #print(c_confi_mask.shape, confi_mask[:, gan:gan+1].shape, gan)
                        #print(l[:, gan:gan+1].max(), l[:, gan:gan+1].min())
                        # cdice = cedice(l[:, gan:gan+1], (c_refine_label == (gan+1)).squeeze(1), uce=False, mask = c_confi_mask[gan:gan+1, 1:])
                        #cdice = cedice(l[:, gan:gan+1], catlas[gan:gan+1], uce=False) #, mask = c_confi_mask[gan:gan+1, 1:])
                        cdice = cedice(l[:, gan:gan+1], (c_refine_label == (gan+1)).squeeze(1), uce=False) #, mask = c_confi_mask[gan:gan+1, 1:])
                        # cdice = edice()
                        #if idx >= 3:
                        dices.append(cdice.detach().cpu().numpy().round(3).item())
                        aux_loss += cdice  / (num_classes - supcount) * weights[idx] * 0.1
                
            # print(dices)
            attns.pop(-1)

            total_loss += dice_loss + aux_loss * aux_weight

            return total_loss, confi_
        else:
            # print("xxk")
            # pass
            return dice_loss, confi_


def get_loss_semi(output, cm, deep_out, target, mask = None, catlas = None, attns = None, pred_teacher = None, label_t = None):
        total_loss = 0.0
        edice = EDiceLoss_partial(output.shape[1])
        num_classes = output.shape[1] - 1
        
        # Starting here it gets spicy!
        dice_loss = edice(output, target.squeeze(1), soft_max = True, mask = mask)

        aux_loss = 0.0
        weights = [0.125, 0.25, 0.5, 1]
        if len(deep_out) != 0:

            for idx, l in enumerate(deep_out):

                aatlas = nn.functional.interpolate(target, l.shape[2:], mode='nearest').float()

                aux_loss +=  edice(l, aatlas.squeeze(1), soft_max = True, mask = mask, uce=False) * weights[idx]
        
        if pred_teacher is not None:

            # print("xxxk")

            refine_label = torch.zeros_like(target, device = target.device)

            refine_output = torch.softmax(pred_teacher, 1)

            #confi = - refine_output.detach() * torch.log(refine_output.detach())
            #confi_mask = confi > 0.3
            confi_mask = torch.logical_or(refine_output > 0.9, refine_output < 0.1).float()

            refine_output = torch.argmax(refine_output, dim = 1).unsqueeze(1)

            for l in range(refine_output.shape[0]):
                if label_t[l]:
                    pass
                else:
                    
                    refine_label[refine_output[l:l+1] == 1] = l+1

            for l in range(refine_output.shape[0]):
                if label_t[l]:
                    refine_label[target == l+1] = l+1
                else:
                    pass

            attns.append(output[:, 1:])
            for idx, l in enumerate(attns):

                c_refine_label = nn.functional.interpolate(refine_label, l.shape[2:], mode='nearest').float()
                c_confi_mask = nn.functional.interpolate(confi_mask, l.shape[2:], mode = 'nearest').float()
                cedice = EDiceLoss_full2(2)

                if idx < 3: 
                    continue
                
                for gan in range(num_classes):
                    if not label_t[gan]:
                #         aux_loss += 
                        aux_loss += cedice(l[:, gan:gan+1], (c_refine_label == (gan+1)).squeeze(1), uce=False, mask = c_confi_mask[:, gan:gan+1]) / 7 * 0.1
            
            attns.pop(-1)
        else:
            for l in attns:
                print("xxk")
                aux_loss += l.mean() * 0

        total_loss += dice_loss + aux_loss

        return total_loss


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def k_one_hot(self, targets:torch.Tensor, n_classes:int, smoothing=0.0):
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                                  .fill_(smoothing /(n_classes-1)) \
                                  .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
        if self.reduction == 'sum' else loss

    def forward(self, inputs, targets):
        assert 0 <= self.smoothing < 1

        targets = self.k_one_hot(targets, inputs.size(-1), self.smoothing)
        log_preds = F.log_softmax(inputs, -1)

        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        return self.reduce_loss(-(targets * log_preds).sum(dim=-1))

def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.shape[0]) # B, 1
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device()).long()
    return SmoothCrossEntropyLoss()(y_pred, y_truth_tensor)
