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
from unet3D import unet3D_with_feam3, get_style_discriminator_output, norm_style_discriminator_output, deep_style_discriminator_output, unet3D_with_deepsup, unet3D_g
from MOTSDataset import AMOSDataSet_newatlas, my_collate

import random
import timeit
import csv
from tensorboardX import SummaryWriter
# from loss_functions.loss_partial import  EDiceLoss_full, EDiceLoss_full2 # EDiceLoss_partial,
from evaluate_amos import predict_sliding, get_dice, get_dice2

from engine import Engine

from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss

from utils import adjust_learning_rate, mask_aug, seedfix
from loss_functions.losses import get_loss_refine, get_loss, SmoothCrossEntropyLoss, bce_loss


start = timeit.default_timer()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():

    parser = argparse.ArgumentParser(description="unet3D_DynConv882")

    parser.add_argument("--data_dir", type=str, default='/apdcephfs_cq10/share_1290796/lh/transoar-main/preprocess/processed_data_f/imagesTr')
    parser.add_argument("--train_list", type=str, default='list/MOTS/MOTS_train.txt')
    parser.add_argument("--val_list", type=str, default='list/MOTS/xx.txt')
    parser.add_argument("--snapshot_dir", type=str, default='snapshots/fold1/')
    parser.add_argument("--reload_path", type=str, default='snapshots/fold1/xx.pth')
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=False)
    parser.add_argument("--input_size", type=str, default='64,64,64')
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--FP16", type=str2bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--itrs_each_epoch", type=int, default=250)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--val_pred_every", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=14)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--weight_std", type=str2bool, default=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_gan", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--is_training", action="store_true")
    parser.add_argument("--random_mirror", type=str2bool, default=True)
    parser.add_argument("--random_scale", type=str2bool, default=True)
    parser.add_argument("--deep_up", type=str2bool, default=False)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--gpu", type=str, default='None')
    parser.add_argument("--disweight", type=float, default=0)
    parser.add_argument("--augmask", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrain_epoch", type = int, default=20)
    return parser


# dice = loss.BinaryDiceLoss_()
# ce = nn.BCEWithLogitsLoss()
# edice = EDiceLoss_partial(14)


def main():

    """Create the model and start the training."""
    parser = get_arguments()
    

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        
        seedfix(args.seed)

        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)

        if args.local_rank == 0:
            writer = SummaryWriter(args.snapshot_dir)
            print(args)

        if not args.gpu == 'None':
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        d, h, w = map(int, args.input_size.split(','))
        input_size = (d, h, w)

        cudnn.benchmark = True
        seed = args.random_seed
        if engine.distributed:
            seed = args.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Create model
        model = unet3D_with_feam3([1, 2, 2, 2, 2], num_classes=args.num_classes, weight_std=args.weight_std, use_cm = [True, True, True], deep_up=args.deep_up)
        model.train()
        refiner = unet3D_g([1, 1, 1, 1, 1], num_classes=2, weight_std=args.weight_std, init_filter=24, in_channel=2) # light weight refiner
        refiner.train()

        if args.deep_up:
            d_style = norm_style_discriminator_output(num_classes=2)
        else:
            d_style = deep_style_discriminator_output(num_classes=2) # num_classes number of input

        device = torch.device('cuda:{}'.format(args.local_rank))
        model.to(device)
        d_style.to(device)
        refiner.to(device)

        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': refiner.parameters()}], lr = args.learning_rate, weight_decay = 0.00005)

        if args.FP16:
            print("Note: Using FP16 during training************")
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

        if args.num_gpus > 1:
            model = engine.data_parallel(model)
            d_style = engine.data_parallel(d_style)
            refiner = engine.data_parallel(refiner)

        # load checkpoint...
        if args.reload_from_checkpoint:
            print('loading from checkpoint: {}'.format(args.reload_path))
            if os.path.exists(args.reload_path):
                if args.FP16:
                    checkpoint = torch.load(args.reload_path, map_location=torch.device('cpu'))
                    model.load_state_dict(checkpoint['model'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    amp.load_state_dict(checkpoint['amp'])
                else:
                    checkpoint = torch.load(args.reload_path, map_location=torch.device('cpu'))
                    model.load_state_dict(checkpoint['model'])
                    refiner.load_state_dict(checkpoint["refiner"])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    d_style.load_state_dict(checkpoint["dis"])
            else:
                print('File not exists in the reload path: {}'.format(args.reload_path))

        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

        trainloader, train_sampler = engine.get_train_loader(
            AMOSDataSet_newatlas(args.data_dir, max_iters=args.itrs_each_epoch * args.batch_size, crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror), 
            collate_fn=my_collate)

        valloader, val_sampler = engine.get_test_loader(
            AMOSDataSet_newatlas(args.data_dir, usage = "valid"))

        best_result = 0
        clist = list(range(13))

        mask_dict = {}
        mask_file = "/apdcephfs_cq10/share_1290796/lh/DoDNet/ours_final/supervise_mask.csv"  # record which organ is labeled in each dataset.
        cfile = open(mask_file, "r")
        reader = csv.reader(cfile)
        for name, mask in reader:
            mask_dict[name] = torch.tensor(np.array(eval(mask)))
        cfile.close()

        ref_loss = []
        output_loss = []

        for epoch in range(args.num_epochs):
            if epoch < args.start_epoch:
                continue

            if engine.distributed:
                train_sampler.set_epoch(epoch)

            d_style.train()

            epoch_loss = []
            adjust_learning_rate(optimizer, epoch, args.learning_rate, args.num_epochs, args.power)

            train_Dice1 = torch.zeros(size=(13, 1)).cuda()  # np.zeros(shape=(7, 2))
            train_senc1 = torch.zeros(size=(13, 1)).cuda()  # np.zeros(shape=(7, 2))
            train_spec1 = torch.zeros(size=(13, 1)).cuda()  # np.zeros(shape=(7, 2))
            count1 = torch.zeros(size=(13, 1)).cuda()  # np.zeros(shape=(7, 2))
            train_Dice2 = torch.zeros(size=(13, 1)).cuda()  # np.zeros(shape=(7, 2))
            train_senc2 = torch.zeros(size=(13, 1)).cuda()  # np.zeros(shape=(7, 2))
            train_spec2 = torch.zeros(size=(13, 1)).cuda()  # np.zeros(shape=(7, 2))
            count2 = torch.zeros(size=(13, 1)).cuda()  # np.zeros(shape=(7, 2))

            for iter, batch in enumerate(trainloader):

                images = torch.from_numpy(batch['image']).cuda()
                if images.shape[2] != 64 or images.shape[3] != 192 or images.shape[4] != 192:
                    continue
                labels = torch.from_numpy(batch['label']).cuda()
                volumeName = "amos_" + batch['name'][0]

                sup_mask = mask_dict[volumeName]

                label_d = torch.tensor(sup_mask[1:]).float()  # for the sup

                task_ids = batch['task_id']

                if int(batch['name'][0]) >= 500:
                    label_t = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]).float()  # for the model index
                else:
                    label_t = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1]).float()

                flist = []
                tlist = []
                dlist = []

                
                for idx, l in enumerate(label_t):
                    if l == 0:
                        flist.append(idx)
                    elif label_d[idx]:
                        tlist.append(idx)
                    if label_d[idx] == 1:
                        dlist.append(idx)

                random.shuffle(flist)
                random.shuffle(tlist)
                random.shuffle(clist)
                random.shuffle(dlist)

                catlas = batch['catlas']
                if catlas[0] is not None:
                    catlas = torch.from_numpy(catlas[0]).cuda()
                else:
                    catlas = None

                cmask = labels.clone()
                for l in range(1,14):
                    if not mask_dict[volumeName][l]:
                        cmask[cmask == l] = 0
                
                # ######################################################################


                optimizer.zero_grad()
                preds, attns, deep_sup, feature_store = model(images, cmask)
                


                if args.local_rank == 0:
                    dices, senc, spec, _  = get_dice(preds.detach(), labels, 1, num_class=args.num_classes - 1)

                    for idx, l in enumerate(dices):
                        if label_t[idx] == 0:
                            train_Dice1[idx, 0] += l
                            train_senc1[idx, 0] += senc[idx]
                            train_spec1[idx, 0] += spec[idx]
                            count1[idx, 0] += 1 

                    #print(dices, np.mean(dices))

                pred_t = torch.softmax(preds.detach().clone(), 1)[0,1:].detach().unsqueeze(1)[tlist]
                if args.augmask:
                    pred_t = mask_aug(pred_t, args.augmask)

                # train the refiner

                if args.augmask > 1:
                    refiner_output = refiner(torch.cat([pred_t, torch.cat([catlas.unsqueeze(1)[tlist] for _ in range(args.augmask)], 0)], 1).float())
                    #refiner_output = refiner(pred_t.float())
                else:
                    refiner_output = refiner(torch.cat([pred_t, catlas.unsqueeze(1)[tlist]], 1).float())
                refine_loss = get_loss_refine(refiner_output, cmask, tlist, args.augmask)
                with torch.no_grad():
                    refiner_output = refiner(torch.cat([torch.softmax(preds, 1)[0, 1:].unsqueeze(1).detach(), catlas.unsqueeze(1) ], 1).float())
                

                if args.local_rank == 0:
                    dices, senc, spec, _  = get_dice2(refiner_output.detach(), labels, 1, num_class = args.num_classes - 1)

                    for idx, l in enumerate(dices):
                        if label_t[idx] == 0:
                            train_Dice2[idx, 0] += l
                            train_senc2[idx, 0] += senc[idx]
                            train_spec2[idx, 0] += spec[idx]
                            count2[idx, 0] += 1

                if epoch < args.pretrain_epoch:
                    term_all, _ = get_loss(preds, 0, [], cmask, [mask_dict[volumeName]], catlas, attns)
                else:

                    if epoch < 50:
                        weight_feature = 0.1 / 50 * epoch
                    else:
                        weight_feature = 0.1

                    term_all, _ = get_loss(preds, 0, [], cmask, [mask_dict[volumeName]], catlas, attns, refiner_output, label_d, weight_feature)

                reduce_all = engine.all_reduce_tensor(term_all)
                reduce_ref_loss = engine.all_reduce_tensor(refine_loss)

                ref_loss.append(refine_loss.detach().cpu().numpy())
                output_loss.append(term_all.detach().cpu().numpy())

                preds = torch.softmax(preds, dim = 1)


                # optimizer_d_style = optim.SGD(d_style.parameters(), lr = float(optimizer.param_groups[0]['lr']), momentum=0.99, nesterov=True)
                optimizer_d_style = optim.Adam(d_style.parameters(), lr = 0.0001)
                adjust_learning_rate(optimizer_d_style, epoch, 0.0001, args.num_epochs, args.power)
                optimizer_d_style.zero_grad()

                d_style.train()
                for param in d_style.parameters():
                    param.requires_grad = False

                # get the discriminator label

                loss_d = 0.

                if not args.deep_up:
                    d_output = d_style(torch.cat([preds[0, 1:].unsqueeze(1), catlas.unsqueeze(1)], 1)[flist].float(), [torch.softmax(l,1)[0][flist].unsqueeze(1) for l in attns])
                    loss_d += bce_loss(d_output, 1)   
                else:
                    weight = [0.125,0.25,0.5,1]
                    attns.append(preds[:, 1:])
                    for idx,l in enumerate(attns[:4]):
                        if idx < 3:
                            continue
                        dis_in = torch.cat([l[0, :].unsqueeze(1), catlas.unsqueeze(1)], 1)[flist].float()
                        d_output = d_style(dis_in)
                        loss_d += bce_loss(d_output, 1) * weight[idx]    
                                
                ####################### Train discriminator networks ######################################
                # enable training mode on discriminator networks
                for param in d_style.parameters():
                    param.requires_grad = True
                
                if not args.deep_up:
                    d_output_ = d_style(torch.cat([preds[0, 1:].unsqueeze(1).detach(), catlas.unsqueeze(1)], 1)[clist].float(), [torch.softmax(l,1)[0][clist].unsqueeze(1).detach() for l in attns])
                    loss_d_ = SmoothCrossEntropyLoss()(d_output_, label_t.reshape(d_output_.shape[0])[clist].to(d_output_.device).long())
                    loss_d__ = engine.all_reduce_tensor(loss_d_)
                else:
                    loss_d_ = 0.
                    weight = [0.125,0.25,0.5,1]

                    for idx,l in enumerate(attns[:4]):
                        if idx < 3:
                            continue
                        dis_in = torch.cat([l[0, :].unsqueeze(1).detach(), catlas.unsqueeze(1)], 1)[clist].float()
                        d_output_ = d_style(dis_in)
                        loss_d_ += SmoothCrossEntropyLoss()(d_output_, label_t.reshape(d_output_.shape[0])[clist].to(d_output_.device).long()) * weight[idx]
                    loss_d__ = engine.all_reduce_tensor(loss_d_)
                    

                if args.FP16:
                    with amp.scale_loss(term_all, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    (term_all + refine_loss + loss_d * args.weight_gan).backward()

                    loss_d_.backward()
                optimizer.step()
                optimizer_d_style.step()


                # get the mask to renew class token in model
                fmask = torch.zeros_like(cmask, device = cmask.device)
                preds_argmask = torch.argmax(preds, 1).unsqueeze(0)


                for l in range(1,14):
                    if mask_dict[volumeName][l]:
                        fmask[torch.logical_and(cmask == l, preds_argmask == l)] = l
                                    
                model.module.renew_token(feature_store, fmask)

                epoch_loss.append(float(reduce_all))

                if (args.local_rank == 0):
                    print(
                        'Epoch {}: {}/{}, lr = {:.4}, loss_seg_Dice = {:.4}, loss_seg_d = {:.4}, loss_Sum = {:.4}, refiner_loss = {:.4}'.format( \
                            epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], term_all.item(),
                            loss_d__.item(), reduce_all.item(), reduce_ref_loss.item()))

            epoch_loss = np.mean(epoch_loss)

            print(np.mean(ref_loss), np.mean(output_loss))


            if args.local_rank == 0:

                count2[count2 == 0] = 1
                train_Dice2 = train_Dice2 / count2
                train_spec2 = train_spec2 / count2
                train_senc2 = train_senc2 / count2

                count1[count1 == 0] = 1
                train_Dice1 = train_Dice1 / count1
                train_spec1 = train_spec1 / count1
                train_senc1 = train_senc1 / count1

                print(train_Dice1[:,0], train_Dice2[:,0])
                print(train_spec1[:,0], train_spec2[:,0])
                print(train_senc1[:,0], train_senc2[:,0])

            if (args.local_rank == 0):
                print(datetime.datetime.now())

            if (args.local_rank == 0):
                print('Epoch_sum {}: lr = {:.4}, loss_Sum = {:.4}'.format(epoch, optimizer.param_groups[0]['lr'],
                                                                          epoch_loss.item()))
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('Train_loss', epoch_loss.item(), epoch)

            if (epoch >= 5) and (args.local_rank == 0) and ((epoch+1) % 50 == 0):
                
                r1, r2 = validate(args, input_size, [model], valloader, args.num_classes, engine)
                model.train()
                c_results = r1+r2
                if c_results > best_result or epoch % 100 == 0:
                    if c_results > best_result:
                        best_result = c_results
                        print("bestresult")
                    print('save model with results ', r1, r2)

                    if args.FP16:
                        checkpoint = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'amp': amp.state_dict()
                        }
                        torch.save(checkpoint, osp.join(args.snapshot_dir, 'MOTS_DynConv_' + args.snapshot_dir.split('/')[-2] + '_e' + str(epoch) + '.pth'))
                    else:
                        checkpoint = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'refiner': refiner.state_dict(),
                            'dis': d_style.state_dict(),
                        }
                        torch.save(checkpoint,osp.join(args.snapshot_dir, 'MOTS_DynConv_' + args.snapshot_dir.split('/')[-2] + '_e' + str(epoch) + '.pth'))

            if (epoch >= args.num_epochs - 1) and (args.local_rank == 0):
                print('save model ...')
                if args.FP16:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict()
                    }
                    torch.save(checkpoint, osp.join(args.snapshot_dir, 'MOTS_DynConv_' + args.snapshot_dir.split('/')[-2] + '_final_e' + str(epoch) + '.pth'))
                else:
                    checkpoint = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'refiner': refiner.state_dict(),
                        }
                    torch.save(checkpoint,osp.join(args.snapshot_dir, 'MOTS_DynConv_' + args.snapshot_dir.split('/')[-2] + '_e' + str(epoch) + '.pth'))
                break

        end = timeit.default_timer()
        print(end - start, 'seconds')


def validate(args, input_size, model, ValLoader, num_classes, engine):

    val_loss = torch.zeros(size=(13, 1)).cuda()  # np.zeros(shape=(7, 1))
    val_Dice1 = torch.zeros(size=(13, 1)).cuda()  # np.zeros(shape=(7, 2))
    val_senc1 = torch.zeros(size=(13, 1)).cuda()  # np.zeros(shape=(7, 2))
    val_spec1 = torch.zeros(size=(13, 1)).cuda()  # np.zeros(shape=(7, 2))
    count1 = torch.zeros(size=(13, 1)).cuda()  # np.zeros(shape=(7, 2))
    val_Dice2 = torch.zeros(size=(13, 1)).cuda()  # np.zeros(shape=(7, 2))
    val_senc2 = torch.zeros(size=(13, 1)).cuda()  # np.zeros(shape=(7, 2))
    val_spec2 = torch.zeros(size=(13, 1)).cuda()  # np.zeros(shape=(7, 2))
    count2 = torch.zeros(size=(13, 1)).cuda()  # np.zeros(shape=(7, 2))
    val_Dice3 = torch.zeros(size=(13, 1)).cuda()  # np.zeros(shape=(7, 2))
    val_senc3 = torch.zeros(size=(13, 1)).cuda()  # np.zeros(shape=(7, 2))
    val_spec3 = torch.zeros(size=(13, 1)).cuda()  # np.zeros(shape=(7, 2))
    count3 = torch.zeros(size=(13, 1)).cuda()  # np.zeros(shape=(7, 2))
    model[0].eval()

    mask_dict = {}
    mask_file = "/apdcephfs_cq10/share_1290796/lh/DoDNet/ours_final/supervise_mask.csv"
    cfile = open(mask_file, "r")
    reader = csv.reader(cfile)
    for name, mask in reader:
        mask_dict[name] = torch.tensor(np.array(eval(mask)))
    cfile.close()

    for index, batch in enumerate(ValLoader):
        # print('%d processd' % (index))
        image, label, name, task_id, affine = batch
        print(index, end = "\r")

        print(name)
        volumeName = "amos_" + name[0]

        sup_mask = mask_dict[volumeName]

        with torch.no_grad():

            pred_logits = predict_sliding(args, model, image.numpy(), input_size, num_classes, task_id)

            # loss = loss_seg_DICE.forward(pred, label) + loss_seg_CE.forward(pred, label)
            loss = torch.tensor(1).cuda()

            dices, senc, spec, _ = get_dice(pred_logits, label, task_id)
            print(dices)

            for idx, l in enumerate(dices):
                if sup_mask[idx+1] == 1:
                    val_Dice3[idx, 0] += l
                    val_senc3[idx, 0] += senc[idx]
                    val_spec3[idx, 0] += spec[idx]
                    count3[idx, 0] += 1

            if int(name[0]) < 510:
                for idx, l in enumerate(dices):
                    val_Dice1[idx, 0] += l
                    val_senc1[idx, 0] += senc[idx]
                    val_spec1[idx, 0] += spec[idx]
                    count1[idx, 0] += 1
            else:
                for idx, l in enumerate(dices):
                    val_Dice2[idx, 0] += l
                    val_senc2[idx, 0] += senc[idx]
                    val_spec2[idx, 0] += spec[idx]
                    count2[idx, 0] += 1
            continue
            print('Task%d-%s Organ:%.4f Tumor:%.4f' % (task_id, name, dice_c1.item(), dice_c2.item()))

            # save
            save_nii(args, pred_sigmoid, label, name, affine)
            print(datetime.datetime.now())

    count1[count1 == 0] = 1
    val_Dice1 = val_Dice1 / count1
    val_spec1 = val_spec1 / count1
    val_senc1 = val_senc1 / count1

    count2[count2 == 0] = 1
    val_Dice2 = val_Dice2 / count2
    val_spec2 = val_spec2 / count2
    val_senc2 = val_senc2 / count2

    count3[count3 == 0] = 1
    val_Dice3 = val_Dice3 / count3
    val_spec3 = val_spec3 / count3
    val_senc3 = val_senc3 / count3

    print(count3, val_Dice3)

    reduce_val_Dice1 = torch.zeros_like(val_Dice1).cuda()
    reduce_val_Dice2 = torch.zeros_like(val_Dice1).cuda()
    for i in range(val_Dice1.shape[0]):
        reduce_val_Dice1[i] = val_Dice1[i]
        reduce_val_Dice2[i] = val_Dice2[i]

    if args.local_rank == 0:
        print("Sum results CT")
        for t in range(13):
            print('Sum: Task%d- Organ:%.4f' % (t, reduce_val_Dice1[t, 0]))
        print("Sum results MRI")
        for t in range(13):
            print('Sum: Task%d- Organ:%.4f' % (t, reduce_val_Dice2[t, 0]))


        print("Sum results CT")
        for t in range(13):
            print('Sum: Task%d- Organ:%.4f' % (t, val_spec1[t, 0]))
        print("Sum results MRI")
        for t in range(13):
            print('Sum: Task%d- Organ:%.4f' % (t, val_spec2[t, 0]))

        print("Sum results CT")
        for t in range(13):
            print('Sum: Task%d- Organ:%.4f' % (t, val_senc1[t, 0]))
        print("Sum results MRI")
        for t in range(13):
            print('Sum: Task%d- Organ:%.4f' % (t, val_senc2[t, 0]))

    return torch.sum(val_Dice3), 0

if __name__ == '__main__':
    main()
