import numpy as np
import os
import SimpleITK as sitk
import glob
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage as ndimage
import random
from scipy.ndimage import gaussian_filter
import copy
import csv

def get_mask_dict_ct(cid):
    mask = [0] * 16  # 16 slots (0=background, 1-15=organs)
    
    # Map case IDs to supervised organs (adjust ranges as needed)
    if int(cid) <= 45:
        mask[4] = 1   # Gall bladder (label 4)
    elif int(cid) <= 85:
        mask[5] = 1   # Esophagus (label 5)
    elif int(cid) <= 135:
        mask[6] = 1   # Liver (label 6)
    elif int(cid) <= 180:
        mask[7] = 1   # Stomach (label 7)
    elif int(cid) <= 242:
        mask[8] = 1   # Aorta (label 8)
    elif int(cid) <= 300:
        mask[9] = 1   # Postcava (label 9)
    elif int(cid) <= 370:
        mask[10] = 1  # Pancreas (label 10)
    elif int(cid) <= 440:
        mask[11] = 1  # Right adrenal gland (label 11)
    elif int(cid) <= 500:
        mask[12] = 1  # Left adrenal gland (label 12)
    elif int(cid) <= 560:
        mask[13] = 1  # Duodenum (label 13)
    elif int(cid) <= 620:
        mask[14] = 1  # Bladder (label 14)
    else:
        mask[15] = 1  # Prostate/uterus (label 15)
    
    return mask[1:]  # Exclude background (keep only labels 1-15)

label_path = "/Users/thurayaalzubaidi/multimodal-PL-1/amos22/labelsTr"
label_files = glob.glob(os.path.join(label_path, "*.nii.gz"))

old_files = copy.deepcopy(label_files)
for l in old_files:
    cname = l.split("/")[-1]
    if "amos" not in cname:
        label_files.remove(l)

random.seed(1)
random.shuffle(label_files)

training_files = label_files[:int(0.7*len(label_files))]

total_shape = [0,0,0]
total = 0

for l in training_files:

    label_nii = sitk.ReadImage(l)
    label_a = sitk.GetArrayFromImage(label_nii)

    total_shape[0] += label_a.shape[0]
    total_shape[1] += label_a.shape[1]
    total_shape[2] += label_a.shape[2]

    #print(label_a.shape)

total_shape[0] /= len(training_files)
total_shape[1] /= len(training_files)
total_shape[2] /= len(training_files)

total_shape = [np.round(l) for l in total_shape]

print(f"The average shape: {total_shape}")

catlas = np.zeros((15, int(total_shape[0]), int(total_shape[1]), int(total_shape[2])))
count = np.zeros((15, 1,1,1)) 

tcount = [24, 25, 25, 21, 64, 64, 63, 64, 79, 25, 79, 79, 79]

import csv
import json

with open("/Users/thurayaalzubaidi/multimodal-PL-1/amos22/dataset.json") as f:
    data = json.load(f)

with open("supervise_mask.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["name", "mask"])
    
    for case in data["training"]:
        case_id = int(case["image"].split("_")[-1].split(".")[0])
        mask = get_mask_dict_ct(case_id)  # Use the updated function
        writer.writerow([case["image"].split("/")[-1], mask])

"""""
mask_dict = {}
mask_file = "/apdcephfs/share_1290796/lh/DoDNet/ours/supervise_mask.csv"
cfile = open(mask_file, "r")
reader = csv.reader(cfile)
for name, mask in reader:
    mask_dict[name] = eval(mask) 

for l in training_files:


    cid = l.split("/")[-1].split("_")[1][:-7]
    # c_sup_mask = mask_dict("amos" + cid)
    c_sup_mask = mask_dict["amos_" + cid][1:]
    
    label_nii = sitk.ReadImage(l)
    label_a = sitk.GetArrayFromImage(label_nii)

    for cidx, gan in enumerate(c_sup_mask):
        if gan:

            if count[cidx] >= tcount[cidx] // 4  + tcount[cidx] % 4:
                continue

            clabel = (label_a == (cidx+1)).astype(np.float32)

            if clabel.sum() == 0:
                continue 

            label_b = ndimage.zoom(clabel, [total_shape[0] / clabel.shape[0], total_shape[1] / clabel.shape[1], total_shape[2] / clabel.shape[2]], order = 0)

            #label_b = gaussian_filter(label_b, sigma=3)

            count[cidx] += 1
            catlas[cidx] += label_b

#count[count == 0] = 1

for gan in range(1, 14):
    if catlas[gan - 1].max() != 0:
        catlas[gan-1] = catlas[gan-1] / catlas[gan-1].max()
        catlas[gan-1] = gaussian_filter(catlas[gan-1], sigma = 3)

print(count)

np.save("atlas_mm_25p.npy", catlas) """




