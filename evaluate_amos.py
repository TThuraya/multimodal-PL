import argparse
import os, sys

sys.path.append("..")

import datetime

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
from scipy.ndimage import gaussian_filter
import SimpleITK as sitk

# from tqdm import tqdm
import os.path as osp

from unet3D import UNet3D, unet3D_with_eam, unet3D_with_eam_baseline, unet3D_with_deepsup, unet3D_with_feam2, unet3D_baseline
# from MOTSDataset import MOTSValDataSet, AMOSDataSet, AMOSDataSet_newatlas, AMOSDataSet_transformed

import random
import timeit
from tensorboardX import SummaryWriter
from loss_functions import loss

from sklearn import metrics
import nibabel as nib
from math import ceil

from engine import Engine
#from apex import amp
import csv
#from apex.parallel import convert_syncbn_model

start = timeit.default_timer()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="MOTS: DynConv solution!")

    parser.add_argument("--data_dir", type=str, default='/apdcephfs/share_1290796/lh/transoar-main/dataset/amos_256_256_128_CT')
    parser.add_argument("--val_list", type=str, default='/list/MOTS/MOTS_test.txt')
    parser.add_argument("--reload_path", type=str, default='snapshots/fold1/MOTS_DynConv_fold1_final_e999.pth')
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=True)
    parser.add_argument("--save_path", type=str, default='outputs/')

    parser.add_argument("--input_size", type=str, default='64,192,192')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--FP16", type=str2bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--val_pred_every", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--weight_std", type=str2bool, default=True)
    parser.add_argument("--print", type=str2bool, default=False)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--dataset_type", type=str, default = "default")

    return parser



def dice_score(preds, labels):  # on GPU
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

    dice = 2 * num / den

    return dice.mean()

def spec_score(preds, labels):  # on GPU
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + 1

    dice = num / den

    return dice.mean()

def senc_score(preds, labels):  # on GPU
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(target, dim=1) + 1

    dice = num / den

    return dice.mean()

def get_dice(preds, labels, t_id, atlas = None, num_class = 13):
    
    preds_r = F.softmax(preds, dim = 1)
    preds = torch.argmax(preds_r, dim = 1)
    #print(preds.shape, labels.shape) 

    dices = []
    senc = []
    spec = []

    if atlas is None:
        for l in range(num_class):
            dices.append(dice_score(preds == (l+1), labels == (l+1)))
            senc.append(senc_score(preds == (l+1), labels == (l+1)))
            spec.append(spec_score(preds == (l+1), labels == (l+1)))
    else:
        for l in range(num_class):
            cpred = (preds == (l+1))
            cpred = (preds_r[:, l+1] + 0.15) > (1-atlas[:, l])
            #torch.logical_and(torch.logical_and(torch.logical_or(cpred, atlas[:, l] > 0.7), atlas[:, l] > 0.), preds_r[:, l+1]>0.1)

            dices.append(dice_score(cpred, labels == (l+1)))
            senc.append(senc_score(cpred, labels == (l+1)))
            spec.append(spec_score(cpred, labels == (l+1)))
        

    return dices, senc, spec, preds

def get_dice2(preds, labels, t_id, atlas = None, num_class = 13):
    
    preds_r = F.softmax(preds, dim = 1)
    preds = torch.argmax(preds_r, dim = 1)
    #print(preds.shape, labels.shape) 

    dices = []
    senc = []
    spec = []

    if atlas is None:
        for l in range(num_class):
            dices.append(dice_score(preds[l:l+1] == 1, labels == (l+1)))
            senc.append(senc_score(preds[l:l+1] == 1, labels == (l+1)))
            spec.append(spec_score(preds[l:l+1] == 1, labels == (l+1)))
    else:
        for l in range(num_class):
            cpred = (preds == (l+1))
            cpred = (preds_r[:, l+1] + 0.15) > (1-atlas[:, l])
            #torch.logical_and(torch.logical_and(torch.logical_or(cpred, atlas[:, l] > 0.7), atlas[:, l] > 0.), preds_r[:, l+1]>0.1)

            dices.append(dice_score(cpred, labels == (l+1)))
            senc.append(senc_score(cpred, labels == (l+1)))
            spec.append(spec_score(cpred, labels == (l+1)))
        

    return dices, senc, spec, preds

def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map

def multi_net(net_list, img, task_id):
    # img = torch.from_numpy(img).cuda()

    padded_prediction = net_list[0](img, task_id)
    #padded_prediction = F.sigmoid(padded_prediction)
    for i in range(1, len(net_list)):
        padded_prediction_i = net_list[i](img, task_id)
        #padded_prediction_i = F.sigmoid(padded_prediction_i)
        padded_prediction += padded_prediction_i
    padded_prediction /= len(net_list)
    return padded_prediction#.cpu().data.numpy()

def predict_sliding(args, net_list, image, tile_size, classes, task_id, tta = False):  # tile_size:32x256x256
    gaussian_importance_map = _get_gaussian(tile_size, sigma_scale=1. / 8)
    
    image_size = image.shape
    overlap = 1 / 4

    strideHW = ceil(tile_size[1] * (1 - overlap))
    strideD = ceil(tile_size[0] * (1 - overlap))
    tile_deps = int(ceil((image_size[2] - tile_size[0]) / strideD) + 1)
    tile_rows = int(ceil((image_size[3] - tile_size[1]) / strideHW) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[4] - tile_size[2]) / strideHW) + 1)
    # print("Need %i x %i x %i prediction tiles @ stride %i x %i px" % (tile_deps, tile_cols, tile_rows, strideD, strideHW))
    full_probs = np.zeros((image_size[0], classes, image_size[2], image_size[3], image_size[4]))#.astype(np.float32)  # 1x4x155x240x240
    count_predictions = np.zeros((image_size[0], classes, image_size[2], image_size[3], image_size[4]))#.astype(np.float32)
    full_probs = torch.from_numpy(full_probs)
    count_predictions = torch.from_numpy(count_predictions)

    for dep in range(tile_deps):
        for row in range(tile_rows):
            for col in range(tile_cols):
                d1 = int(dep * strideD)
                x1 = int(col * strideHW)
                y1 = int(row * strideHW)
                d2 = min(d1 + tile_size[0], image_size[2])
                x2 = min(x1 + tile_size[2], image_size[4])
                y2 = min(y1 + tile_size[1], image_size[3])
                d1 = max(int(d2 - tile_size[0]), 0)
                x1 = max(int(x2 - tile_size[2]), 0)  # for portrait images the x1 underflows sometimes
                y1 = max(int(y2 - tile_size[1]), 0)  # for very few rows y1 underflows

                img = image[:, :, d1:d2, y1:y2, x1:x2]
                img = torch.from_numpy(img).cuda()

                prediction1 = multi_net(net_list, img, task_id)

                ## tta
                if tta:
                    prediction2 = torch.flip(multi_net(net_list, torch.flip(img, [2]), task_id), [2])
                    prediction3 = torch.flip(multi_net(net_list, torch.flip(img, [3]), task_id), [3])
                    prediction4 = torch.flip(multi_net(net_list, torch.flip(img, [4]), task_id), [4])
                    prediction5 = torch.flip(multi_net(net_list, torch.flip(img, [2,3]), task_id), [2,3])
                    prediction6 = torch.flip(multi_net(net_list, torch.flip(img, [2,4]), task_id), [2,4])
                    prediction7 = torch.flip(multi_net(net_list, torch.flip(img, [3,4]), task_id), [3,4])
                    prediction8 = torch.flip(multi_net(net_list, torch.flip(img, [2,3,4]), task_id), [2,3,4])
                    prediction = (prediction1 + prediction2 + prediction3 + prediction4 + prediction5 + prediction6 + prediction7 + prediction8) / 8.
                
                    prediction = prediction.cpu()
                else:
                    prediction = prediction1.cpu()

                prediction[:,:] *= gaussian_importance_map

                if isinstance(prediction, list):
                    shape = np.array(prediction[0].shape)
                    shape[0] = prediction[0].shape[0] * len(prediction)
                    shape = tuple(shape)
                    preds = torch.zeros(shape).cuda()
                    bs_singlegpu = prediction[0].shape[0]
                    for i in range(len(prediction)):
                        preds[i * bs_singlegpu: (i + 1) * bs_singlegpu] = prediction[i]
                    count_predictions[:, :, d1:d2, y1:y2, x1:x2] += 1
                    full_probs[:, :, d1:d2, y1:y2, x1:x2] += preds

                else:
                    count_predictions[:, :, d1:d2, y1:y2, x1:x2] += gaussian_importance_map
                    full_probs[:, :, d1:d2, y1:y2, x1:x2] += prediction

    full_probs /= count_predictions
    return full_probs

def save_nii(args, pred, label, name, affine): # bs, c, WHD
    seg_pred_2class = np.asarray(np.around(pred), dtype=np.uint8)
    seg_pred_0 = seg_pred_2class[:, 0, :, :, :]
    seg_pred_1 = seg_pred_2class[:, 1, :, :, :]
    seg_pred = np.zeros_like(seg_pred_0)
    if name[0][0:3]!='spl':
        seg_pred = np.where(seg_pred_0 == 1, 1, seg_pred)
        seg_pred = np.where(seg_pred_1 == 1, 2, seg_pred)
    else:# spleen only organ
        seg_pred = seg_pred_0

    label_0 = label[:, 0, :, :, :]
    label_1 = label[:, 1, :, :, :]
    seg_label = np.zeros_like(label_0)
    seg_label = np.where(label_0 == 1, 1, seg_label)
    seg_label = np.where(label_1 == 1, 2, seg_label)

    if name[0][0:3]!='cas':
        seg_pred = seg_pred.transpose((0, 2, 3, 1))
        seg_label = seg_label.transpose((0, 2, 3, 1))

    # save
    for tt in range(seg_pred.shape[0]):
        seg_pred_tt = seg_pred[tt]
        seg_label_tt = seg_label[tt]
        seg_pred_tt = nib.Nifti1Image(seg_pred_tt, affine=affine[tt])
        seg_label_tt = nib.Nifti1Image(seg_label_tt, affine=affine[tt])
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        seg_label_save_p = os.path.join(args.save_path + '/%s_label.nii.gz' % (name[tt]))
        seg_pred_save_p = os.path.join(args.save_path + '/%s_pred.nii.gz' % (name[tt]))
        nib.save(seg_label_tt, seg_label_save_p)
        nib.save(seg_pred_tt, seg_pred_save_p)
    return None

def validate(args, input_size, model, ValLoader, num_classes, engine, usage = "train"):
    # Remove .cuda() from tensor initializations
    val_loss = torch.zeros(size=(13, 1))
    val_Dice1 = torch.zeros(size=(13, 1))
    val_senc1 = torch.zeros(size=(13, 1))
    val_spec1 = torch.zeros(size=(13, 1))
    count1 = torch.zeros(size=(13, 1))
    val_Dice2 = torch.zeros(size=(13, 1))
    val_senc2 = torch.zeros(size=(13, 1))
    val_spec2 = torch.zeros(size=(13, 1))
    count2 = torch.zeros(size=(13, 1))
    val_Dice3 = torch.zeros(size=(13, 1))
    val_senc3 = torch.zeros(size=(13, 1))
    val_spec3 = torch.zeros(size=(13, 1))
    count3 = torch.zeros(size=(13, 1))
    
    # Replace any torch.cuda.* calls
    model[0].eval()
    
    # Update device assignments
    device = torch.device('cpu')
    
    file = open(os.path.join("/apdcephfs_cq10/share_1290796/lh/DoDNet/ours_final/snapshots/", args.reload_path.split("/")[-2], args.reload_path.split("/")[-1] + "_valie.csv"), "w")
    print(os.path.join("/apdcephfs_cq10/share_1290796/lh/DoDNet/ours_final/snapshots/", args.reload_path.split("/")[-2], args.reload_path.split("/")[-1] + "_valie.csv"))
    writer = csv.writer(file)

    index_ct = 0
    index_mri = 0


    for index, batch in enumerate(ValLoader):
        # print('%d processd' % (index))
        image, label, name, task_id, affine = batch

        # if int(name[0]) != 551: # 89 for ct
        #     continue
        print(index, end = "\r")
        with torch.no_grad():
            
            pred_logits = predict_sliding(args, model, image.numpy(), input_size, num_classes, task_id)

            # loss = loss_seg_DICE.forward(pred, label) + loss_seg_CE.forward(pred, label)
            loss = torch.tensor(1)

            dices, senc, spec, preds = get_dice(pred_logits, label, task_id)
            print(dices)

            print([l.numpy().item() for l in dices])
            #print(senc)
            #print(spec)

            #dices, senc, spec, preds = get_dice(pred_logits, label, task_id, affine)
            #print(dices)
            #print(senc)
            #print(spec)

            writer.writerow([name[0], *[l.numpy().item() for l in dices]])

            if int(name[0]) < 507:
                for idx, l in enumerate(dices):
                    val_Dice1[idx] += l
                    val_senc1[idx] += senc[idx]
                    val_spec1[idx] += spec[idx]
                    count1[idx] += 1

                    dice_record_ct[index_ct, idx] = l.numpy()
                index_ct += 1
            else:
                for idx, l in enumerate(dices):
                    val_Dice2[idx] += l
                    val_senc2[idx] += senc[idx]
                    val_spec2[idx] += spec[idx]
                    count2[idx] += 1

                    dice_record_mri[index_mri, idx] = l.numpy()
                index_mri += 1

            
            if args.print and False:
                store_path = os.path.join("/apdcephfs_cq10/share_1290796/lh/DoDNet/ours_final/testlog/", args.reload_path.split("/")[-2])
                if not os.path.exists(store_path):
                    os.mkdir(store_path)
                cpath = os.path.join(store_path, name[0])
                # pred_logits = F.softmax(pred_logits, dim = 1)


                for l in range(num_classes - 1):
                    cpre = (preds == (l+1)).cpu().float().numpy()
                    clabel = (label == (l+1)).float().numpy()
                    #print(cpre.shape) # 1 128 256 256
                    #print(clabel.shape) # 1 1 128 256 256
                    #print(affine.shape) # 1 13 128 256 256
                    t_index = np.argmax(clabel.sum((0,1,3,4)))
                    

                    plt.subplot(7, 6, l*3 + 1)
                    plt.axis("off")
                    plt.imshow(image.numpy()[0, 0, t_index], cmap = "gray"  )

                    plt.subplot(7, 6, l*3 + 2)
                    plt.axis("off")
                    plt.imshow(clabel[0, 0, t_index], cmap = "gray")


                    plt.subplot(7, 6, l*3 + 3)
                    plt.axis("off")
                    plt.imshow(cpre[0, t_index], cmap = "gray")
                    
                    #plt.subplot(7, 10, l*3 + 3)
                    #plt.axis("off")
                    #plt.imshow(affine.numpy()[0, l, t_index], cmap = "gray")

                    #plt.subplot(7, 10, l*5 + 4)
                    ##plt.axis("off")
                    #plt.imshow(pred_logits.cpu().numpy()[0, l+1, t_index], cmap = "gray")

                    #plt.subplot(7, 10, l*5 + 5)
                    #plt.axis("off")
                    #plt.imshow(image.numpy()[0, 0, t_index], cmap = "gray")  

                    # print(pred_logits[0, l+1, t_index].max(), pred_logits[0, l+1, t_index].min(), pred_logits[0, l+1, t_index].mean())

                plt.subplots_adjust(left=0.46, bottom=0, right=None, top=None, wspace=0.02, hspace=0.02)
                plt.savefig(cpath + ".png", dpi = 200)

            if args.print:
                store_path = os.path.join("/apdcephfs_cq10/share_1290796/lh/DoDNet/ours_final/testlog/", args.reload_path.split("/")[-2])
                if not os.path.exists(store_path):
                    os.mkdir(store_path)
                cpath = os.path.join(store_path, name[0])
                # pred_logits = F.softmax(pred_logits, dim = 1)
                t_index = 60
                print(image.shape, label.shape, preds.shape)

                pred_c = preds.clone()
                
                label[label > 10] = 0
                pred_c[pred_c > 10] = 0
            
                label[label == 5] = 0
                pred_c[pred_c == 5] = 0

                label[label == 4] = 0
                pred_c[pred_c == 4] = 0

                label[label == 2] = 4
                pred_c[pred_c == 2] = 4
                
                label = label * 15
                pred_c = pred_c * 15

                plt.subplot(1, 3, 1)
                plt.axis("off")
                plt.imshow(image.numpy()[0, 0, t_index], cmap = "gray")

                plt.subplot(1, 3, 2)
                plt.axis("off")
                plt.imshow(label.numpy()[0, 0, t_index], vmin = 0, vmax = 150, cmap = "nipy_spectral")

                plt.subplot(1, 3, 3)
                plt.axis("off")
                plt.imshow(pred_c.numpy()[0, t_index], vmin = 0, vmax = 150, cmap = "nipy_spectral")

                plt.subplots_adjust(left=0, bottom=0, right=None, top=None, wspace=0.02, hspace=0.02)
                plt.savefig(cpath + "60.png", dpi = 200)
                print(cpath)

                preds = preds.numpy()
                out = sitk.GetImageFromArray(preds)
                sitk.WriteImage(out, cpath + ".nii.gz")

            continue
            print('Task%d-%s Organ:%.4f Tumor:%.4f' % (task_id, name, dice_c1.item(), dice_c2.item()))

            # save
            save_nii(args, pred_sigmoid, label, name, affine)
            print(datetime.datetime.now())

    file.close()
    count1[count1 == 0] = 1
    val_Dice1 = val_Dice1 / count1
    val_spec1 = val_spec1 / count1
    val_senc1 = val_senc1 / count1

    count2[count2 == 0] = 1
    val_Dice2 = val_Dice2 / count2
    val_spec2 = val_spec2 / count2
    val_senc2 = val_senc2 / count2

    print(count1, count2)

    print(np.mean(dice_record_ct[:index_ct], 0), np.std(dice_record_ct[:index_ct], 0))
    print(np.mean(dice_record_mri[:index_mri], 0), np.std(dice_record_mri[:index_mri], 0))

    reduce_val_Dice1 = torch.zeros_like(val_Dice1)
    reduce_val_Dice2 = torch.zeros_like(val_Dice1)
    for i in range(val_Dice1.shape[0]):
        reduce_val_Dice1[i] = val_Dice1[i]
        reduce_val_Dice2[i] = val_Dice2[i]

    if args.local_rank == 0:
        print("Sum results CT")
        for t in range(13):
            print('Sum: Task%d- Organ:%.4f' % (t, reduce_val_Dice1[t]))

        print("mean_result", reduce_val_Dice1.mean())
        print("Sum results MRI")
        for t in range(13):
            print('Sum: Task%d- Organ:%.4f' % (t, reduce_val_Dice2[t]))
        print("mean_result", reduce_val_Dice2.mean())


        print("Sum results CT")
        for t in range(13):
            print('Sum: Task%d- Organ:%.4f' % (t, val_spec1[t]))
        print("Sum results MRI")
        for t in range(13):
            print('Sum: Task%d- Organ:%.4f' % (t, val_spec2[t]))

        print("Sum results CT")
        for t in range(13):
            print('Sum: Task%d- Organ:%.4f' % (t, val_senc1[t]))
        print("Sum results MRI")
        for t in range(13):
            print('Sum: Task%d- Organ:%.4f' % (t, val_senc2[t]))

    return
    return reduce_val_loss.mean(), reduce_val_Dice




def main():
    """Create the model and start the training."""
    parser = get_arguments()
    print(parser)

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)

        d, h, w = map(int, args.input_size.split(','))
        input_size = (d, h, w)

        cudnn.benchmark = True
        seed = 1234
        if engine.distributed:
            seed = args.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Create network. 
        if "fdeep" in args.reload_path or True:
            model = unet3D_with_feam2([1, 2, 2, 2, 2], num_classes=args.num_classes, weight_std=args.weight_std)
            # model = unet3D_baseline([1, 2, 2, 2, 2], num_classes=args.num_classes, weight_std=args.weight_std)
        elif "deep" not in args.reload_path:
            model = unet3D_with_eam([1, 2, 2, 2, 2], num_classes=args.num_classes, weight_std=args.weight_std)
        else:
            model = unet3D_with_deepsup([1, 2, 2, 2, 2], num_classes=args.num_classes, weight_std=args.weight_std, use_cm = [True, True, True], deep_up = True)

        args_usage = "test"

        model = nn.DataParallel(model)

        model.eval()

        device = torch.device('cpu')
        model.to(device)

        if args.num_gpus > 1:
            model = engine.data_parallel(model)

        # load checkpoint...
        if args.reload_from_checkpoint:
            print('loading from checkpoint: {}'.format(args.reload_path))
            if os.path.exists(args.reload_path):
                if args.FP16:
                    checkpoint = torch.load(args.reload_path, map_location=torch.device('cpu'))
                    model.load_state_dict(checkpoint['model'])
                    # optimizer.load_state_dict(checkpoint['optimizer'])
                    # amp.load_state_dict(checkpoint['amp'])
                else:
                    checkpoint = torch.load(args.reload_path, map_location=torch.device('cpu'))
                    model.load_state_dict(checkpoint["model"])
                    #refiner.load_state_dict(checkpoint["refiner"])
                    #optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('File not exists in the reload path: {}'.format(args.reload_path))
        '''
        print(model.module.class_token2.shape)
        token = model.module.class_token2
        a = torch.zeros((token.shape[0], token.shape[0]))

        for l1 in range(token.shape[0]):
            for l2 in range(token.shape[0]):
                a[l1, l2] = nn.CosineSimilarity(dim = 0)(token[l1], token[l2])
        print(a)
        print(token)

        exit(0)
        '''

        if args.dataset_type == "default":
            valloader, val_sampler = engine.get_test_loader(
                AMOSDataSet(args.data_dir, usage = "test"))
        else:
            valloader, val_sampler = engine.get_test_loader(
                AMOSDataSet_newatlas("/apdcephfs_cq10/share_1290796/lh/transoar-main/preprocess/processed_data_f/imagesTr", usage = args_usage))

        print('validate ...')
        validate(args, input_size, [model], valloader, args.num_classes, engine, args_usage)

        end = timeit.default_timer()
        print(end - start, 'seconds')


if __name__ == '__main__':
    main()
