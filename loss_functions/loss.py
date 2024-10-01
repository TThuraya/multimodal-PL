import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.ndimage as nd
from matplotlib import pyplot as plt
from torch import Tensor, einsum


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + self.smooth

        dice_score = 2*num / den
        dice_loss = 1 - dice_score

        dice_loss_avg = dice_loss[target[:,0]!=-1].sum() / dice_loss[target[:,0]!=-1].shape[0]

        return dice_loss_avg

class BinaryDiceLoss_(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss_, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + self.smooth

        dice_score = 2*num / den

        #print(dice_score, num, den)

        dice_loss = 1 - dice_score

        #dice_loss_avg = dice_loss[target[:,0]!=-1].sum() / dice_loss[target[:,0]!=-1].shape[0]

        return dice_loss


class DiceLoss4MOTS(nn.Module):
    def __init__(self, weight=None, ignore_index=None, num_classes=3, **kwargs):
        super(DiceLoss4MOTS, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.dice = BinaryDiceLoss(**self.kwargs)

    def forward(self, predict, target, sigmoid = True):
        
        total_loss = []
        if sigmoid:
            predict = F.sigmoid(predict)

        for i in range(self.num_classes):
            if i != self.ignore_index:
                dice_loss = self.dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == self.num_classes, \
                        'Expect weight shape [{}], get[{}]'.format(self.num_classes, self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss.append(dice_loss)

        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss==total_loss]

        return total_loss.sum()/total_loss.shape[0]


class CELoss4MOTS(nn.Module):
    def __init__(self, ignore_index=None, num_classes=3, **kwargs):
        super(CELoss4MOTS, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def weight_function(self, mask):
        weights = torch.ones_like(mask).float()
        voxels_sum = mask.shape[0] * mask.shape[1] * mask.shape[2]
        for i in range(2):
            voxels_i = [mask == i][0].sum().cpu().numpy()
            w_i = np.log(voxels_sum / voxels_i).astype(np.float32)
            weights = torch.where(mask == i, w_i * torch.ones_like(weights).float(), weights)

        return weights

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        total_loss = []
        for i in range(self.num_classes):
            if i != self.ignore_index:
                ce_loss = self.criterion(predict[:, i], target[:, i])
                ce_loss = torch.mean(ce_loss, dim=[1,2,3])
                ce_loss_avg = ce_loss[target[:, i, 0, 0, 0] != -1].sum() / ce_loss[target[:, i, 0, 0, 0] != -1].shape[0]
                total_loss.append(ce_loss_avg)

        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss == total_loss]
        return total_loss.sum()/total_loss.shape[0]


class BCELoss(nn.Module):
    def __init__(self, ignore_index=None, num_classes=4, sigmoid=True, **kwargs):
        super(BCELoss, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        if sigmoid:
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.criterion = nn.BCELoss(reduction='none')

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.num_classes):
            temp_prob = input_tensor == i+1  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, predict, target):
        target = self._one_hot_encoder(target)
        
        assert predict.shape == target.shape, 'predict & target shape do not match'
        loss = self.criterion(predict, target) 
        # loss = loss.sum(dim=1)
        return loss.mean()#torch.masked_select(loss, target.sum(dim=1) != 0).mean()


# class BCELossNoBG(nn.Module):
#     def __init__(self, ignore_index=None, num_classes=12, **kwargs):
#         super(BCELossNoBG, self).__init__()
#         self.kwargs = kwargs
#         self.num_classes = num_classes
#         self.ignore_index = ignore_index
#         self.criterion = nn.BCEWithLogitsLoss(reduction='none')
#         self.task_nbg = {0:[1,2],
#                         1:[3,4],
#                         2:[5,6],
#                         3:[7,8],
#                         4:[9],
#                         5:[10],
#                         6:[11]}

#     def _one_hot_encoder(self, input_tensor):
#         tensor_list = []
#         for i in range(self.num_classes):
#             temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
#             tensor_list.append(temp_prob.unsqueeze(1))
#         output_tensor = torch.cat(tensor_list, dim=1)
#         return output_tensor.float()

#     def forward(self, predict, target, task_id):
#         loss = 0.
#         # target = self._one_hot_encoder(target)
#         for i in range(predict.size(0)):
#             for j in self.task_nbg[task_id[i]]:
#                 loss += self.criterion(predict[i:i+1,j,...], (target[i:i+1,...]==j).float()).mean([1,2,3])/len(self.task_nbg[task_id[i]])
#         return loss

class BCELossNoBG5(nn.Module):
    def __init__(self, ignore_index=None, num_classes=4, **kwargs):
        super(BCELossNoBG5, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.task_nbg = {0:1,
                         1:2,
                         3:3,
                         6:4}

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.num_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, predict, target, task_id):
        loss = 0.
        for i in range(predict.size(0)):
            loss += self.criterion(predict[i:i+1,self.task_nbg[task_id[i]],...],
                                    (target[i:i+1,...]==self.task_nbg[task_id[i]]).float()).mean()
        return loss/predict.size(0)

# class ObjLoss(nn.Module):
#     def __init__(self, **kwargs):
#         super(ObjLoss, self).__init__()
#         self.kwargs = kwargs
#         self.task_nbg = {0:[1,2],
#                         1:[3,4],
#                         2:[5,6],
#                         3:[7,8],
#                         4:[9],
#                         5:[10],
#                         6:[11]}

#     def forward(self, obj, target, task):
#         loss = 0.
#         for i in range(obj.size(0)):
#             obj_ = torch.log(obj[i,self.task_nbg[task[i]],...].sum(0))
#             obj_[target[i]==0] = 0.
#             loss = loss - obj_.mean()/obj.size(0)
#         return loss

# class RecLoss(nn.Module):
#     def __init__(self, **kwargs):
#         super(RecLoss, self).__init__()
#         self.kwargs = kwargs

#     def forward(self, pred, gt, label, sigma2):
#         sigma2_map = torch.zeros(sigma2.size(0), sigma2.size(1), label.size(-3), label.size(-2), label.size(-1)).cuda()
#         for b in range(sigma2.size(0)):
#             sigma2_map[b,...] = sigma2[b,:,label[b,...]]
#         return torch.mean((pred-gt)**2/sigma2_map)

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target) #torch.sum(target)#
        z_sum = torch.sum(score * score) #torch.sum(score)#

        loss = (2 * intersect) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        # class_wise_dice = []
        loss = 0.0
        for i in range(1, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            # class_wise_dice.append(1.0 - dice.item())
            loss += dice# * weight[i]
        return loss / (self.n_classes-1)


class DiceLoss2(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss2, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i+1  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target) #torch.sum(target)#
        z_sum = torch.sum(score * score) #torch.sum(score)#

        loss = (2 * intersect) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, sigmoid=True):
        if sigmoid:
            inputs = torch.sigmoid(inputs)
        target = self._one_hot_encoder(target)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        loss = 0.0
        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice
        return loss / self.n_classes

class TAL(nn.Module):
    def __init__(self, ignore_index=None, voxels=64*192*192, norm=False):
        super(TAL, self).__init__()
        self.ignore_index = ignore_index
        self.norm=norm
        self.task_bg = {0:[0,3,4,5,6,7,8,9,10,11],
                        1:[0,1,2,5,6,7,8,9,10,11],
                        2:[0,1,2,3,4,7,8,9,10,11],
                        3:[0,1,2,3,4,5,6,9,10,11],
                        4:[0,1,2,3,4,5,6,7,8,10,11],
                        5:[0,1,2,3,4,5,6,7,8,9,11],
                        6:[0,1,2,3,4,5,6,7,8,9,10]}
        self.task_nbg = {0:[1,2],
                        1:[3,4],
                        2:[5,6],
                        3:[7,8],
                        4:[9],
                        5:[10],
                        6:[11]}
        self.voxels = voxels
        self.voxel_avg = torch.zeros(12)
        self.voxel_sum = torch.zeros(12)
        self.voxel_count = torch.zeros(12)
        self.weights = torch.ones(1,12).cuda()

    def update_weights(self, val, dim):
        self.voxel_count[dim] += 1 
        self.voxel_sum[dim] += val/self.voxels
        self.voxel_avg[dim] = self.voxel_sum[dim] / self.voxel_count[dim]
        self.weights[dim] = torch.log(1./self.voxel_avg[dim])

    def forward(self, inputs, targets, task_id, softmax=True):
        loss = 0.
        if self.norm:
            weights = self.weights.repeat(inputs.size(0),1)/self.weights.mean()
        else:
            weights = self.weights.repeat(inputs.size(0),1)
        for i in range(inputs.size(0)):
            bg_ids = self.task_bg[task_id[i]]
            nbg_ids = self.task_nbg[task_id[i]]
            inputs_1 = inputs[i, bg_ids, :, :, :].clone().sum(0, keepdims=True)
            inputs_2 = inputs[i, nbg_ids, :, :, :].clone()
            inputs_ = torch.cat([inputs_1, inputs_2], 0).unsqueeze(0)
            if task_id[i] <= 4:
                delta = -2.0*task_id[i]
            else:
                delta = -task_id[i] - 4.0
            targets[i,...] = torch.where(targets[i,...]>0, delta, 0.) + targets[i,...]
            if softmax:
                loss += F.cross_entropy(inputs_, targets[[i]], weight=weights[i, [0]+nbg_ids])
            else:
                loss += F.nll_loss(torch.log(inputs_), targets[[i]])

        return loss 


class TAL6(nn.Module):
    def __init__(self, ignore_index=None):
        super(TAL6, self).__init__()
        self.ignore_index = ignore_index
        self.task_bg = {0:[0,2,3,4,5],
                        1:[0,1,3,4,5],
                        2:[0,1,2,4,5],
                        3:[0,1,2,3,5],
                        6:[0,1,2,3,4]}
        self.task_nbg = {0:[1],
                        1:[2],
                        2:[3],
                        3:[4],
                        6:[5]}

    def forward(self, inputs, targets, task_id, softmax=True):
        loss = 0.
        for i in range(inputs.size(0)):
            bg_ids = self.task_bg[task_id[i]]
            nbg_ids = self.task_nbg[task_id[i]]
            inputs_1 = inputs[i, bg_ids, :, :, :].clone().sum(0, keepdims=True)
            inputs_2 = inputs[i, nbg_ids, :, :, :].clone()
            inputs_ = torch.cat([inputs_1, inputs_2], 0).unsqueeze(0)
            # if task_id[i] <= 3:
            #     delta = task_id[i] + 1.0
            # else:
            #     delta = task_id[i] - 1.0
            targets[i,...] = torch.where(targets[i,...]>0, 1., 0.) 
            if softmax:
                loss += F.cross_entropy(inputs_, targets[[i]])
            else:
                loss += F.nll_loss(torch.log(inputs_), targets[[i]])
                
        return loss 

class TAL5(nn.Module):
    def __init__(self, ignore_index=None):
        super(TAL5, self).__init__()
        self.ignore_index = ignore_index
        self.task_bg = {0:[0,2,3,4],
                        1:[0,1,3,4],
                        3:[0,1,2,4],
                        6:[0,1,2,3]}
        self.task_nbg = {0:[1],
                        1:[2],
                        3:[3],
                        6:[4]}

    def forward(self, inputs, targets, task_id, softmax=True):
        loss = 0.
        for i in range(inputs.size(0)):
            bg_ids = self.task_bg[task_id[i]]
            nbg_ids = self.task_nbg[task_id[i]]
            inputs_1 = inputs[i, bg_ids, :, :, :].clone().sum(0, keepdims=True)
            inputs_2 = inputs[i, nbg_ids, :, :, :].clone()
            inputs_ = torch.cat([inputs_1, inputs_2], 0).unsqueeze(0)
            targets[i,...] = torch.where(targets[i,...]>0, 1., 0.) 
            if softmax:
                loss += F.cross_entropy(inputs_, targets[[i]])
            else:
                loss += F.nll_loss(torch.log(inputs_), targets[[i]])
                
        return loss 

class MargExcLoss(nn.Module):
    def __init__(self):
        super(MargExcLoss, self).__init__()
        # self.n_classes = n_classes
        self.task_nbg = {0:[0,1,2],
                        1:[0,3,4],
                        2:[0,5,6],
                        3:[0,7,8],
                        4:[0,9],
                        5:[0,10],
                        6:[0,11]}

    def _dice_loss(self, score, target, inv=False):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target, dim=[1,2,3])
        y_sum = torch.sum(target * target, dim=[1,2,3])
        z_sum = torch.sum(score * score, dim=[1,2,3])
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        if inv:
            loss = loss.sum()
        else:
            loss = (1 - loss).sum()
        return loss

    def _one_hot_encoder(self, input_tensor, classes):
        tensor_list = []
        for i in classes:
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(0))
        output_tensor = torch.cat(tensor_list, dim=0)
        return output_tensor.float()

    def forward(self, inputs, targets, task_id):
        inputs = torch.softmax(inputs, 1)
        # Marginal Loss
        loss_marg_dice = 0.
        loss_marg_CE = 0.
        loss_exc_dice = 0.
        loss_exc_CE = 0.
        for i in range(inputs.size(0)):
            nbg_ids = self.task_nbg[task_id[i]]
            inputs_marg = inputs[i,nbg_ids,...].unsqueeze(0)
            if task_id[i] <= 4 :
                delta = -2.0*task_id[i]
            else:
                delta = -task_id[i] - 4.0
            targets_marg = (torch.where(targets[i,...]>0, delta, 0.) + targets[i,...]).unsqueeze(0)
            
            loss_marg_CE += F.cross_entropy(inputs_marg, targets_marg.long())
            target_i = self._one_hot_encoder(targets[i], nbg_ids)

            loss_marg_dice += self._dice_loss(inputs_marg[0], target_i)
        # Exclusive loss
            target_e = 1 - self._one_hot_encoder(targets[i], list(range(12)))
            target_e[:,0] = 0.
            loss_exc_dice += self._dice_loss(inputs[i], target_e, True)
            loss_exc_CE += (torch.log(inputs[i]+1)*target_e).mean([-1,-2,-3]).sum()


        return loss_marg_dice/inputs.size(0), loss_marg_CE/inputs.size(0), loss_exc_dice/inputs.size(0), loss_exc_CE/inputs.size(0)


