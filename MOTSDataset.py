import os
import os.path as osp
import numpy as np
import random
import collections
import pickle
import torch
import torch.nn as nn
import torchvision
import cv2
from torch.utils import data
import matplotlib.pyplot as plt
try:
    import SimpleITK as sitk
except ImportError:
    print("Please install SimpleITK: pip install SimpleITK")
import scipy.ndimage as ndimage
import glob
import copy
import math
import csv
from batchgenerators.transforms.abstract_transforms import Compose as BatchGeneratorsCompose
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, \
    BrightnessTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
try:
    import nibabel as nib
except ImportError:
    print("Please install nibabel: pip install nibabel")

def get_train_transform():
    tr_transforms = []

    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1, data_key="image"))
    tr_transforms.append(
        GaussianBlurTransform(blur_sigma=(0.5, 1.), different_sigma_per_channel=True, p_per_channel=0.5,
                              p_per_sample=0.2, data_key="image"))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.75, 1.25), p_per_sample=0.15, data_key="image"))
    tr_transforms.append(BrightnessTransform(0.0, 0.1, True, p_per_sample=0.15, p_per_channel=0.5, data_key="image"))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15, data_key="image"))
    #tr_transforms.append(
    #    SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5, order_downsample=0,
    #                                   order_upsample=3, p_per_sample=0.25,
    #                                   ignore_axes=None, data_key="image"))
    #tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, retain_stats=True,
    #                                    p_per_sample=0.15, data_key="image"))

    # now we compose these transforms together
    tr_transforms = BatchGeneratorsCompose(tr_transforms)
    return tr_transforms

def my_collate(batch):
    image, label, name, task_id, catlas = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    task_id = np.stack(task_id, 0)

    #if catlas[0] is not None:
    #    print(catlas[0].shape, "asss")

    data_dict = {'image': image, 'label': label, 'name': name, 'task_id': task_id, "catlas": catlas, 'image_r': image.copy()}
    tr_transforms = get_train_transform()
    data_dict = tr_transforms(**data_dict)
    return data_dict


class AMOSDataSet_newatlas(data.Dataset):

    # root = "/apdcephfs/share_1290796/lh/transoar-main/preprocess/processed_data/imagesTr" # labelsTr

    def __init__(self, root, max_iters=None, crop_size=(64, 256, 256), mean=(128, 128, 128), scale=False,
                 mirror=False, ignore_label=255, usage = "train", use_ct_mri = [True, True]):
        self.root = root
        #self.list_path = list_path
        self.crop_d, self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.usage = usage

        allfiles = sorted(glob.glob(os.path.join(root, "*.nii.gz")))

        old_files = copy.deepcopy(allfiles)
        for l in old_files:
            cname = l.split("/")[-1]
            if "amos" not in cname:
                allfiles.remove(l)

        # print(allfiles) # check if need sorted

        random.seed(1)
        random.shuffle(allfiles)

        # data split
        if self.usage == "train":
            self.files = allfiles[:int(0.7*len(allfiles))]
        elif self.usage == "valid":
            self.files = allfiles[int(0.7*len(allfiles)):int(0.8*len(allfiles))]
        else:
            self.files = allfiles[int(0.8*len(allfiles)):]

        # data filter
        if not use_ct_mri[0]:
            old_files = copy.deepcopy(self.files)
            for l in old_files:
                cname = l.split("/")[-1]
                if int(cname) < 410:
                    self.files.remove(l)
        if not use_ct_mri[1]:
            old_files = copy.deepcopy(self.files)
            for l in old_files:
                cname = l.split("/")[-1]
                if int(cname) >= 410:
                    self.files.remove(l)

        atlas_path = "atlas_mm.npy"
        # atlas load
        self.atlas = np.load(atlas_path) # 13 128 256 256
        print(atlas_path)
        self.cores = []

        for gan in range(self.atlas.shape[0]):
            cgan = self.atlas[gan]
            aranges = []
            for l2 in cgan.shape:
                aranges.append(torch.arange(l2))

            mesh = torch.meshgrid(*aranges)

            c_cores = []

            for dim in range(len(cgan.shape)):
                c_core = mesh[dim]
                c_core = c_core[cgan > 0].float().mean().int()
                c_cores.append(c_core)

            self.cores.append(c_cores)

        # load mask
        self.mask_dict = {}
        mask_file = "/apdcephfs_cq10/share_1290796/lh/DoDNet/ours_final/supervise_mask.csv"
        cfile = open(mask_file, "r")
        reader = csv.reader(cfile)
        for name, mask in reader:
            self.mask_dict[name] = torch.tensor(np.array(eval(mask)))
        cfile.close()

        if self.usage == "train":
            old_files = copy.deepcopy(self.files)
            for datafile in old_files:
                print(datafile)
                name = datafile.replace("images", "labels").replace("_0000", "").split("/")[-1][5:-7]
                c_supervise_mask = self.mask_dict["amos_"+name]
                if c_supervise_mask[1]:
                    task_idd = 2
                elif c_supervise_mask[5]:
                    task_idd = 0
                else:
                    task_idd = 1
                

        print('{} images are loaded!'.format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def truncate(self, CT, task_id):
        min_HU = -325
        max_HU = 325
        subtract = 0
        divide = 325.
        
        # truncate
        if float(task_id) < 500:
            CT[np.where(CT <= min_HU)] = min_HU
            CT[np.where(CT >= max_HU)] = max_HU
            CT = CT - subtract
            CT = CT / divide
        else:
            CT = CT - np.mean(CT)
            CT = CT / np.std(CT)
        return CT

    def id2trainId(self, label, task_id):
        if task_id == 0 or task_id == 1 or task_id == 3:
            organ = (label >= 1)
            tumor = (label == 2)
        elif task_id == 2:
            organ = (label == 1)
            tumor = (label == 2)
        elif task_id == 4 or task_id == 5:
            organ = None
            tumor = (label == 1)
        elif task_id == 6:
            organ = (label == 1)
            tumor = None
        else:
            print("Error, No such task!")
            return None

        shape = label.shape
        results_map = np.zeros((2, shape[0], shape[1], shape[2])).astype(np.float32)

        if organ is None:
            results_map[0, :, :, :] = results_map[0, :, :, :] - 1
        else:
            results_map[0, :, :, :] = np.where(organ, 1, 0)
        if tumor is None:
            results_map[1, :, :, :] = results_map[1, :, :, :] - 1
        else:
            results_map[1, :, :, :] = np.where(tumor, 1, 0)

        return results_map

    def locate_bbx(self, label, scaler):

        scale_d = int(self.crop_d * scaler)
        scale_h = int(self.crop_h * scaler)
        scale_w = int(self.crop_w * scaler)

        img_h, img_w, img_d = label.shape
        boud_h, boud_w, boud_d = np.where(label >= 1)
        margin = 32  # pixels
        bbx_h_min = boud_h.min()
        bbx_h_max = boud_h.max()
        bbx_w_min = boud_w.min()
        bbx_w_max = boud_w.max()
        bbx_d_min = boud_d.min()
        bbx_d_max = boud_d.max()
        if (bbx_h_max - bbx_h_min) <= scale_h:
            bbx_h_maxt = bbx_h_max + (scale_h - (bbx_h_max - bbx_h_min)) // 2
            bbx_h_mint = bbx_h_min - (scale_h - (bbx_h_max - bbx_h_min)) // 2
            bbx_h_max = bbx_h_maxt
            bbx_h_min = bbx_h_mint
        if (bbx_w_max - bbx_w_min) <= scale_w:
            bbx_w_maxt = bbx_w_max + (scale_w - (bbx_w_max - bbx_w_min)) // 2
            bbx_w_mint = bbx_w_min - (scale_w - (bbx_w_max - bbx_w_min)) // 2
            bbx_w_max = bbx_w_maxt
            bbx_w_min = bbx_w_mint
        if (bbx_d_max - bbx_d_min) <= scale_d:
            bbx_d_maxt = bbx_d_max + (scale_d - (bbx_d_max - bbx_d_min)) // 2
            bbx_d_mint = bbx_d_min - (scale_d - (bbx_d_max - bbx_d_min)) // 2
            bbx_d_max = bbx_d_maxt
            bbx_d_min = bbx_d_mint
        bbx_h_min = np.max([bbx_h_min - margin, 0])
        bbx_h_max = np.min([bbx_h_max + margin, img_h])
        bbx_w_min = np.max([bbx_w_min - margin, 0])
        bbx_w_max = np.min([bbx_w_max + margin, img_w])
        bbx_d_min = np.max([bbx_d_min - margin, 0])
        bbx_d_max = np.min([bbx_d_max + margin, img_d])

        if random.random() < 0.8:
            d0 = random.randint(bbx_d_min, bbx_d_max - scale_d)
            h0 = random.randint(bbx_h_min, bbx_h_max - scale_h)
            w0 = random.randint(bbx_w_min, bbx_w_max - scale_w)
        else:
            d0 = random.randint(0, img_d - scale_d)
            h0 = random.randint(0, img_h - scale_h)
            w0 = random.randint(0, img_w - scale_w)
        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w
        return [h0, h1, w0, w1, d0, d1]

    def pad_image(self, img, target_size):
        """Pad an image up to the target size."""
        rows_missing = math.ceil(target_size[0] - img.shape[0])
        cols_missing = math.ceil(target_size[1] - img.shape[1])
        dept_missing = math.ceil(target_size[2] - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image2(self, img, target_size):
        """Pad an image up to the target size."""
        rows_missing = math.ceil(target_size[0] - img.shape[1])
        cols_missing = math.ceil(target_size[1] - img.shape[2])
        dept_missing = math.ceil(target_size[2] - img.shape[3])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0,0), (0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def __getitem__(self, index):
        datafiles = self.files[index]
        # read nii file
    
        image = sitk.GetArrayFromImage(sitk.ReadImage(datafiles))
        label = sitk.GetArrayFromImage(sitk.ReadImage(datafiles.replace("images", "labels").replace("_0000", "")))
        name = datafiles.replace("images", "labels").replace("_0000", "").split("/")[-1][5:-7]
        
        # tidx = 0

        '''
        c_supervise_mask = self.mask_dict["amos_" + name]
        
        for idx, gan in enumerate(c_supervise_mask):  # use the first label to do the registration
            if gan == 1:
                tidx == idx
                break 
        

        
        clabel = (label == tidx).astype(np.float32)
        aranges = []
        for l2 in clabel.shape:
            aranges.append(torch.arange(l2))


        mesh = torch.meshgrid(*aranges)

        #print(name, image.shape, self.atlas.shape)

        c_cores = []

        for dim in range(len(clabel.shape)):
            c_core = mesh[dim]
            c_core = c_core[clabel > 0].float().mean().int()
            c_cores.append(c_core)
        catlas = np.zeros((13, *image.shape))

        atlas_cores = self.cores[tidx - 1]

        rrange = [[0, self.atlas.shape[1]], [0, self.atlas.shape[2]], [0, self.atlas.shape[3]]]
        trange = [[0, clabel.shape[0]], [0, clabel.shape[1]], [0, clabel.shape[2]]]
        for idx in range(len(atlas_cores)):
            if c_cores[idx] - atlas_cores[idx] <= 0:
                shift = atlas_cores[idx] - c_cores[idx] 
                length = np.min([clabel.shape[idx], self.atlas.shape[idx + 1]-shift])
                rrange[idx] = [shift, shift + length]
                trange[idx] = [0, length]
            else:
                shift = c_cores[idx] - atlas_cores[idx]
                length = np.min([clabel.shape[idx]-shift, self.atlas.shape[idx + 1]])
                rrange[idx] = [0, length]
                trange[idx] = [shift, length+shift]


        catlas[:,trange[0][0]:trange[0][1], trange[1][0]:trange[1][1], trange[2][0]:trange[2][1]] = self.atlas[:,rrange[0][0]:rrange[0][1], rrange[1][0]:rrange[1][1], rrange[2][0]:rrange[2][1]]
        '''

        catlas = nn.functional.interpolate(torch.tensor(self.atlas).unsqueeze(0), image.shape).numpy()[0] # just resize the atlas to image's shape. rigid registeration.

        if image.shape != label.shape:
            print("fxxk", name, image.shape, label.shape, datafiles)
            final_shape = [
                min([image.shape[0], label.shape[0]]),
                min([image.shape[1], label.shape[1]]),
                min([image.shape[2], label.shape[2]])
            ]
            image = image[:final_shape[0], :final_shape[1], :final_shape[2]]
            label = label[:final_shape[0], :final_shape[1], :final_shape[2]]


        image = self.pad_image(image, [self.crop_h+5, self.crop_w+5, self.crop_d+5])
        label = self.pad_image(label, [self.crop_h+5, self.crop_w+5, self.crop_d+5])
        catlas = self.pad_image2(catlas, [self.crop_h+5, self.crop_w+5, self.crop_d+5])

        image = self.truncate(image, name)

        # simply crop:
        if self.usage == "train":
            b = np.random.randint(label.shape[0] - self.crop_h)
            c = np.random.randint(label.shape[1] - self.crop_w)
            a = np.random.randint(label.shape[2] - self.crop_d)
            image = image[b:b+self.crop_h, c:c+self.crop_w, a:a + self.crop_d]
            label = label[b:b+self.crop_h, c:c+self.crop_w, a:a + self.crop_d]
            catlas = catlas[:, b:b + self.crop_h, c:c+self.crop_w, a:a+self.crop_d]
        else:
            catlas = catlas

        image = image[np.newaxis, :]
        label = label[np.newaxis, :]

        image = image.transpose((0, 3, 1, 2))  # Channel x Depth x H x W
        label = label.transpose((0, 3, 1, 2))  # Depth x H x W
        catlas = catlas.transpose((0, 3, 1, 2))

        image = image.astype(np.float32)
        label = label.astype(np.float32)

        return image.copy(), label.copy(), name, name, catlas

class AMOSDataSet_newatlas_onlyct(data.Dataset):

    # root = "/apdcephfs/share_1290796/lh/transoar-main/preprocess/processed_data/imagesTr" # labelsTr

    def __init__(self, root, max_iters=None, crop_size=(64, 256, 256), mean=(128, 128, 128), scale=False,
                 mirror=False, ignore_label=255, usage = "train", usedataset = ["amos_ct"], only_data = -1):
        self.root = root
        #self.list_path = list_path
        self.crop_d, self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.usage = usage

        allfiles = glob.glob(os.path.join(root, "*.nii.gz"))

        old_files = copy.deepcopy(allfiles)
        for l in old_files:
            cname = l.split("/")[-1]
            if "amos" not in cname:
                allfiles.remove(l)

        random.seed(2)
        random.shuffle(allfiles)

        
        if self.usage == "train":
            self.files = allfiles[:int(0.7*len(allfiles))]
        elif self.usage == "valid":
            self.files = allfiles[int(0.7*len(allfiles)):int(0.8*len(allfiles))]
        else:
            self.files = allfiles[int(0.8*len(allfiles)):]

        count = [0,0,0,0]
        for l in self.files:
            cname = l.split("/")[-1]
            if "amos" in cname:
                cid = cname.split("_")[1]
                if int(cid) <= 410:
                    count[0] += 1
                else:
                    count[1] += 1
            elif "CHAOS" in cname:
                count[2] += 1
            elif "img" in cname:
                count[3] += 1

        if "amos_ct" not in usedataset:
            old_files = copy.deepcopy(self.files)
            for l in old_files:
                cname = l.split("/")[-1]
                
                if "amos" not in cname:
                    continue
                
                cid = cname.split("_")[1]
                if int(cid) <= 410:
                    self.files.remove(l)
                    count[0] -= 1
                 
        if "amos_mri" not in usedataset:
            old_files = copy.deepcopy(self.files)
            for l in old_files:
                cname = l.split("/")[-1]
                
                if "amos" not in cname:
                    continue
                
                cid = cname.split("_")[1]
                if int(cid) > 410:
                    self.files.remove(l)
                    count[1] -= 1
        if "chaos" not in usedataset:
            old_files = copy.deepcopy(self.files)
            for l in old_files:
                cname = l.split("/")[-1]

                if "CHAOS" in cname:
                    self.files.remove(l)
                    count[2] -= 1

        if "msd" not in usedataset:
            old_files = copy.deepcopy(self.files)
            for l in old_files:
                cname = l.split("/")[-1]

                if "img" in cname:
                    self.files.remove(l)
                    count[3] -= 1


        

        #if self.usage == "train":
        #atlas_path = "/apdcephfs/share_1290796/lh/transoar-main/atlas_only_ct.npy"
        atlas_path = "atlas_only_ct.npy"
        #atlas_path = "/apdcephfs/share_1290796/lh/transoar-main/atlas_only_ct_5.npy"
        #atlas_path = "/apdcephfs/share_1290796/lh/transoar-main/atlas_only_ct_1.npy"

        self.atlas = np.load(atlas_path) # 13 128 256 256
        print(atlas_path)
        print(self.atlas.shape)
        self.cores = []

        for gan in range(self.atlas.shape[0]):
            cgan = torch.tensor(self.atlas[gan])
            aranges = []
            for l2 in cgan.shape:
                aranges.append(torch.arange(l2))

            mesh = torch.meshgrid(*aranges)

            c_cores = []

            for dim in range(len(cgan.shape)):

                c_core = ((mesh[dim][cgan > 0] * cgan[cgan > 0].float()).sum() / cgan.sum()).int()
                c_cores.append(c_core)

            self.cores.append(c_cores)
        '''
        if self.usage == "train":
            old_files = copy.deepcopy(self.files)
            for l in old_files:
                label = sitk.GetArrayFromImage(sitk.ReadImage(l.replace("images", "labels").replace("_0000", "")))
                cname = l.split("/")[-1]
                c_supervise_mask = get_mask_dict_ct(cname)
                for gan in range(len(c_supervise_mask)):
                    if c_supervise_mask[gan]:
                        if (label == gan).sum() == 0:
                            self.files.remove(l)
        '''

        if self.usage == "train":
            old_files = copy.deepcopy(self.files)
            for datafiles in old_files:
                cname = datafiles.split("/")[-1]
                c_supervise_mask = get_mask_dict_ct_re(cname.split("_")[1])
                if only_data != -1 and c_supervise_mask[only_data] != 1:
                    self.files.remove(datafiles)

        print('{} images are loaded!'.format(len(self.files)), count)


        print("Cores of atlas: ", self.cores)
        # load mask
        

    def __len__(self):
        return len(self.files)

    def truncate(self, img, task_id):
        min_HU = -325
        max_HU = 325
        subtract = 0
        divide = 325.
        
        # truncate
        if float(task_id) > 0.5: # CT
            img[np.where(img <= min_HU)] = min_HU
            img[np.where(img >= max_HU)] = max_HU
            img = img - subtract
            img = img / divide
        else: # MRI
            img = img - np.mean(img)
            img = img / np.std(img)
        return img

    def id2trainId(self, label, task_id):
        if task_id == 0 or task_id == 1 or task_id == 3:
            organ = (label >= 1)
            tumor = (label == 2)
        elif task_id == 2:
            organ = (label == 1)
            tumor = (label == 2)
        elif task_id == 4 or task_id == 5:
            organ = None
            tumor = (label == 1)
        elif task_id == 6:
            organ = (label == 1)
            tumor = None
        else:
            print("Error, No such task!")
            return None

        shape = label.shape
        results_map = np.zeros((2, shape[0], shape[1], shape[2])).astype(np.float32)

        if organ is None:
            results_map[0, :, :, :] = results_map[0, :, :, :] - 1
        else:
            results_map[0, :, :, :] = np.where(organ, 1, 0)
        if tumor is None:
            results_map[1, :, :, :] = results_map[1, :, :, :] - 1
        else:
            results_map[1, :, :, :] = np.where(tumor, 1, 0)

        return results_map

    def locate_bbx(self, label, scaler):

        scale_d = int(self.crop_d * scaler)
        scale_h = int(self.crop_h * scaler)
        scale_w = int(self.crop_w * scaler)

        img_h, img_w, img_d = label.shape
        boud_h, boud_w, boud_d = np.where(label >= 1)
        margin = 32  # pixels
        bbx_h_min = boud_h.min()
        bbx_h_max = boud_h.max()
        bbx_w_min = boud_w.min()
        bbx_w_max = boud_w.max()
        bbx_d_min = boud_d.min()
        bbx_d_max = boud_d.max()
        if (bbx_h_max - bbx_h_min) <= scale_h:
            bbx_h_maxt = bbx_h_max + (scale_h - (bbx_h_max - bbx_h_min)) // 2
            bbx_h_mint = bbx_h_min - (scale_h - (bbx_h_max - bbx_h_min)) // 2
            bbx_h_max = bbx_h_maxt
            bbx_h_min = bbx_h_mint
        if (bbx_w_max - bbx_w_min) <= scale_w:
            bbx_w_maxt = bbx_w_max + (scale_w - (bbx_w_max - bbx_w_min)) // 2
            bbx_w_mint = bbx_w_min - (scale_w - (bbx_w_max - bbx_w_min)) // 2
            bbx_w_max = bbx_w_maxt
            bbx_w_min = bbx_w_mint
        if (bbx_d_max - bbx_d_min) <= scale_d:
            bbx_d_maxt = bbx_d_max + (scale_d - (bbx_d_max - bbx_d_min)) // 2
            bbx_d_mint = bbx_d_min - (scale_d - (bbx_d_max - bbx_d_min)) // 2
            bbx_d_max = bbx_d_maxt
            bbx_d_min = bbx_d_mint
        bbx_h_min = np.max([bbx_h_min - margin, 0])
        bbx_h_max = np.min([bbx_h_max + margin, img_h])
        bbx_w_min = np.max([bbx_w_min - margin, 0])
        bbx_w_max = np.min([bbx_w_max + margin, img_w])
        bbx_d_min = np.max([bbx_d_min - margin, 0])
        bbx_d_max = np.min([bbx_d_max + margin, img_d])

        if random.random() < 0.8:
            d0 = random.randint(bbx_d_min, bbx_d_max - scale_d)
            h0 = random.randint(bbx_h_min, bbx_h_max - scale_h)
            w0 = random.randint(bbx_w_min, bbx_w_max - scale_w)
        else:
            d0 = random.randint(0, img_d - scale_d)
            h0 = random.randint(0, img_h - scale_h)
            w0 = random.randint(0, img_w - scale_w)
        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w
        return [h0, h1, w0, w1, d0, d1]

    def pad_image(self, img, target_size):
        """Pad an image up to the target size."""
        rows_missing = math.ceil(target_size[0] - img.shape[0])
        cols_missing = math.ceil(target_size[1] - img.shape[1])
        dept_missing = math.ceil(target_size[2] - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def pad_image2(self, img, target_size):
        """Pad an image up to the target size."""
        rows_missing = math.ceil(target_size[0] - img.shape[1])
        cols_missing = math.ceil(target_size[1] - img.shape[2])
        dept_missing = math.ceil(target_size[2] - img.shape[3])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0,0), (0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def __getitem__(self, index):
        datafiles = self.files[index]
        # read nii file
    
        image = sitk.GetArrayFromImage(sitk.ReadImage(datafiles))
        
        cname = datafiles.split("/")[-1]
        name = cname
        ctype = 0 # 0 mri, 1 ct
        if "amos" in cname:

            label = sitk.GetArrayFromImage(sitk.ReadImage(datafiles.replace("images", "labels").replace("_0000", "")))
            #label[label>=9] = 0

            cid = cname.split("_")[1]
            
            if int(cid) < 410:
                ctype = 1
            else:
                ctype = 0
        elif "CHAOS" in cname:
            label = sitk.GetArrayFromImage(sitk.ReadImage(datafiles.replace("images", "labels")))  
            label = convert_seg_chao(label) 
        else:
            label = sitk.GetArrayFromImage(sitk.ReadImage(datafiles.replace("images", "labels").replace("img", "label")))  
            label = convert_seg_msd(label) 
            ctype = 1

        c_supervise_mask = get_mask_dict_ct(cname) # bug

        if image.shape != label.shape:
            print("fxxk", name, image.shape, label.shape, datafiles)
            final_shape = [
                min([image.shape[0], label.shape[0]]),
                min([image.shape[1], label.shape[1]]),
                min([image.shape[2], label.shape[2]])
            ]
            image = image[:final_shape[0], :final_shape[1], :final_shape[2]]
            label = label[:final_shape[0], :final_shape[1], :final_shape[2]]

        ############################################# atlas import and alignment
        if self.usage == "train" and False:

            all_cores = []
            sup_gan_idxs = []
            aranges = []
            
            for l2 in label.shape:
                aranges.append(torch.arange(l2))
            mesh = torch.meshgrid(*aranges)

            for idx, l in enumerate(c_supervise_mask):
                if l:
                    clabel = (label == idx).astype(np.float32)

                    c_cores = []

                    for dim in range(len(clabel.shape)):
                        c_core = mesh[dim]
                        c_core = c_core[clabel > 0].float().mean().int()
                        c_cores.append(c_core)

                    if c_cores[0] > clabel.shape[0] or c_cores[0] < 0:
                        
                        continue

                    all_cores.append(c_cores)  
                    sup_gan_idxs.append(idx)

            catlas = np.zeros((8, *image.shape))

            rrange = [[0, self.atlas.shape[1]], [0, self.atlas.shape[2]], [0, self.atlas.shape[3]]]
            trange = [[0, clabel.shape[0]], [0, clabel.shape[1]], [0, clabel.shape[2]]]
            for idx in range(3):
                
                try:
                    all_shift = []
                    for idxx, candi in enumerate(sup_gan_idxs):
                        all_shift.append(all_cores[idxx][idx] - self.cores[candi-1][idx])
                    cshift = int(np.mean(all_shift))
                except:
                    cshift = 0

                if cshift <= 0:
                    shift = -cshift
                    length = np.min([image.shape[idx], self.atlas.shape[idx + 1]-shift])
                    rrange[idx] = [shift, shift + length]
                    trange[idx] = [0, length]
                else:
                    shift = cshift
                    length = np.min([image.shape[idx]-shift, self.atlas.shape[idx + 1]])
                    rrange[idx] = [0, length]
                    trange[idx] = [shift, length+shift]

            #print(rrange, trange, catlas.shape, self.atlas.shape, clabel.shape, catlas[:, trange[0][0]:trange[0][1], trange[1][0]:trange[1][1], trange[2][0]:trange[2][1]].shape, self.atlas[:, rrange[0][0]:rrange[0][1], rrange[1][0]:rrange[1][1], rrange[2][0]:rrange[2][1]].shape)
            catlas[:, trange[0][0]:trange[0][1], trange[1][0]:trange[1][1], trange[2][0]:trange[2][1]] = self.atlas[:, rrange[0][0]:rrange[0][1], rrange[1][0]:rrange[1][1], rrange[2][0]:rrange[2][1]]

        else:

            catlas = nn.functional.interpolate(torch.tensor(self.atlas).unsqueeze(0), image.shape).numpy()[0]

        #print(label.shape, image.shape, label.max()) # 256 256 128

        


        image = self.pad_image(image, [self.crop_h+5, self.crop_w+5, self.crop_d+5])
        label = self.pad_image(label, [self.crop_h+5, self.crop_w+5, self.crop_d+5])
        catlas = self.pad_image2(catlas, [self.crop_h+5, self.crop_w+5, self.crop_d+5])

        #print(name, catlas.shape, image.shape)

        image = self.truncate(image, ctype)

        # simply crop:
        if self.usage == "train":
            b = np.random.randint(label.shape[0] - self.crop_h)
            c = np.random.randint(label.shape[1] - self.crop_w)
            a = np.random.randint(label.shape[2] - self.crop_d)
            image = image[b:b+self.crop_h, c:c+self.crop_w, a:a + self.crop_d]
            label = label[b:b+self.crop_h, c:c+self.crop_w, a:a + self.crop_d]
            catlas = catlas[:, b:b + self.crop_h, c:c+self.crop_w, a:a+self.crop_d]


        #print(name, catlas.shape, image.shape)

        image = image[np.newaxis, :]
        label = label[np.newaxis, :]

        image = image.transpose((0, 3, 1, 2))  # Channel x Depth x H x W
        label = label.transpose((0, 3, 1, 2))  # Depth x H x W
        catlas = catlas.transpose((0, 3, 1, 2))

        image = image.astype(np.float32)
        label = label.astype(np.float32)

        return image.copy(), label.copy(), name, name, catlas


