from torch import nn, Tensor
import torch
from torch.nn.modules.loss import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from torch.cuda.amp import autocast
import numpy as np
import torch.nn.functional as F



class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        #print(tensor_list[0].shape)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, mask):
        target = target.float()

        score = score[mask.bool()]
        target = target[mask.squeeze(1).bool()]

        smooth = 1e-5
        intersect = torch.sum(score * target )
        y_sum = torch.sum(target*target) 
        z_sum = torch.sum(score*score) # score * score ?
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=True, mask = None):
        #print(inputs.shape, target.shape)
        #print(target.max(), target.min())

        target = self._one_hot_encoder(target)
        #print(inputs.max(), inputs.min())
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            if mask is None:

                dice = self._dice_loss(inputs[:, i], target[:, i], torch.ones_like(target[:, i], device = target.device))
            else:
                dice = self._dice_loss(inputs[:, i], target[:, i], mask[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class EDiceLoss_partial(nn.Module):
    """Dice loss tailored to Brats need.
    """

    def __init__(self, n_classes):
        super(EDiceLoss_partial, self).__init__()
        self.labels = ["ET", "TC", "WT", 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a']
        self.device = "cpu"
        self.n_classes = n_classes
        self.diceloss = DiceLoss(n_classes = n_classes)
        self.bce = BCELoss()

    def forward(self, inputs, target, mask = None, soft_max = True, uce = True):
        dice = 0
        ce = 0
        if soft_max:
            inputs_softmax = torch.softmax(inputs, dim=1)    
        else:
            inputs_softmax = torch.sigmoid(inputs)   
        # print(torch.sum(inputs, 1).mean())
        # print(inputs.shape, target.shape)
        if mask is None:
            mask = []
            for l in range(inputs.shape[0]):
                mask.append(torch.ones(self.n_classes))
        
        

        dice += self.diceloss(inputs_softmax, target, softmax=False, weight = mask[0])
        
        if uce:
            with autocast(enabled=False):
                for l in range(inputs.shape[1]):
                    ce += self.bce(inputs_softmax[:, l].float(), (target[:]==l).float()) * mask[0][l]

        #print(ce, dice, inputs_softmax.max(), inputs_softmax.min())

        if uce:
            return dice + ce
        else:
            return dice


class EDiceLoss_full(nn.Module):
    """Dice loss tailored to Brats need.
    """

    def __init__(self, n_classes):
        super(EDiceLoss_full, self).__init__()
        self.labels = ["ET", "TC", "WT", 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a']
        self.device = "cpu"
        self.n_classes = n_classes
        self.diceloss = DiceLoss(n_classes = n_classes)
        self.bce = BCELoss()
        self.mce = CrossEntropyLoss()

    def forward(self, inputs, target, logits = "softmax", uce=True):
        dice = 0
        ce = 0
        if logits == "softmax":
            inputs_softmax = torch.softmax(inputs, dim=1)       
        elif logits == "extend":
            inputs_softmax = torch.cat([torch.softmax(inputs, 1)[:,1:8], torch.sigmoid(inputs)[:,8:]], 0)
        else:
            inputs_softmax = torch.sigmoid(inputs)    
             
        #print(torch.sum(inputs, 1).mean())

        dice += self.diceloss(inputs_softmax, target, softmax=False)
      
        ce += self.mce(inputs.float(), target.long())

        # print(ce, dice, inputs_softmax.max(), inputs_softmax.min())
        if uce:
            return dice + ce
        else:
            return dice

class EDiceLoss_full2(nn.Module):
    """Dice loss tailored to Brats need. for binary
    """

    def __init__(self, n_classes):
        super(EDiceLoss_full2, self).__init__()
        self.labels = ["ET", "TC", "WT", 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a']
        self.device = "cpu"
        self.n_classes = n_classes
        self.diceloss = DiceLoss(n_classes = n_classes)
        self.bce = BCEWithLogitsLoss()
        self.mce = CrossEntropyLoss()

    def forward(self, inputs, target, uce=True, mask = None, sigmoid = True):
        dice = 0
        ce = 0
        if sigmoid:
            inputs_sigmoid = torch.sigmoid(inputs)
        else:
            inputs_sigmoid = inputs       
        #print(torch.sum(inputs, 1).mean())

        if mask is None:
            dice += self.diceloss._dice_loss(inputs_sigmoid, target, torch.ones_like(target, device = target.device).unsqueeze(0))

        else:
            dice += self.diceloss._dice_loss(inputs_sigmoid, target, mask)
      
        # print(ce, dice, inputs_softmax.max(), inputs_softmax.min())
        if uce:
            ce += self.bce(inputs.float().squeeze(0), target.float())
            return dice + ce
        else:
            return dice

def linear_combination(x, y, epsilon): 
    return epsilon*x + (1-epsilon)*y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon:float=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss/n, nll, self.epsilon)