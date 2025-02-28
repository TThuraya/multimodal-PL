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
import json

def get_mask_dict_ct(cid):
    """
    Generate supervision mask for semi-supervised learning.
    Labels 1-3 are intentionally unsupervised.
    
    Args:
        cid: Case ID (int)
            < 500: CT data
            >= 500: MRI data
    """
    mask = [0] * 16  # 16 slots (0=background, 1-15=organs)
    
    # Only process CT data (id < 500)
    cid = int(cid)
    if cid >= 500:  # MRI data
        return [0] * 15  # Return all zeros for MRI data
        
    # Supervised organs only (labels 4-15)
    # Adjusted ranges based on 200 CT training images
    if cid <= 45:
        mask[4] = 1   # Gall bladder
    elif cid <= 85:
        mask[5] = 1   # Esophagus
    elif cid <= 135:
        mask[6] = 1   # Liver
    elif cid <= 180:
        mask[7] = 1   # Stomach
    elif cid <= 242:
        mask[8] = 1   # Aorta
    elif cid <= 300:
        mask[9] = 1   # Postcava
    elif cid <= 370:
        mask[10] = 1  # Pancreas
    elif cid <= 440:
        mask[11] = 1  # Right adrenal gland
    elif cid <= 460:
        mask[12] = 1  # Left adrenal gland
    elif cid <= 480:
        mask[13] = 1  # Duodenum
    elif cid <= 500:
        mask[14] = 1  # Bladder
    # Note: mask[15] (Prostate/uterus) remains 0 as it might be specific to certain cases
    
    return mask[1:]  # Exclude background (keep only labels 1-15)

def generate_supervision_mask(json_path="amos/dataset.json"):
    """Generate supervision mask from dataset.json"""
    with open(json_path) as f:
        data = json.load(f)
    
    with open("supervise_mask.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["name", "mask"])
        
        for case in data["training"]:
            case_id = int(case["image"].split("_")[-1].split(".")[0])
            mask = get_mask_dict_ct(case_id)  # Using existing function
            writer.writerow([case["image"].split("/")[-1], mask])

def generate_atlas(data_dir="/Users/thurayaalzubaidi/multimodal-PL/amos/labelsTr"):
    """Generate atlas from training labels"""
    print("\n=== Starting Atlas Generation ===")
    
    # First generate the supervision mask
    print("Step 1: Generating supervision mask...")
    generate_supervision_mask()
    print("✓ Supervision mask generated")
    
    # Get all label files
    print("\nStep 2: Looking for label files...")
    label_files = glob.glob(os.path.join(data_dir, "*.nii.gz"))
    if not label_files:
        print(f"Current working directory: {os.getcwd()}")
        print(f"Looking in directory: {data_dir}")
        print(f"Directory contents: {os.listdir(data_dir) if os.path.exists(data_dir) else 'Directory not found'}")
        raise ValueError(f"No .nii.gz files found in {data_dir}")

    print(f"✓ Found {len(label_files)} label files")
    
    # Calculate average shape from training files
    print("\nStep 3: Calculating average shape...")
    random.seed(1)
    random.shuffle(label_files)
    training_files = label_files[:int(0.7*len(label_files))]
    print(f"Using {len(training_files)} files for training")
    
    total_shape = [0,0,0]
    for i, l in enumerate(training_files):
        if i % 10 == 0:  # Print progress every 10 files
            print(f"Processing file {i+1}/{len(training_files)} for shape calculation")
        label_nii = sitk.ReadImage(l)
        label_a = sitk.GetArrayFromImage(label_nii)
        total_shape[0] += label_a.shape[0]
        total_shape[1] += label_a.shape[1]
        total_shape[2] += label_a.shape[2]

    total_shape = [s/len(training_files) for s in total_shape]
    total_shape = [int(np.round(s)) for s in total_shape]
    print(f"✓ Average shape calculated: {total_shape}")

    # Generate atlas
    print("\nStep 4: Generating atlas...")
    catlas = np.zeros((15, total_shape[0], total_shape[1], total_shape[2]))
    count = np.zeros((15, 1, 1, 1))

    # Process each training file
    total_files = len(training_files)
    for file_idx, l in enumerate(training_files):
        print(f"Processing file {file_idx+1}/{total_files}: {os.path.basename(l)}")
        label_nii = sitk.ReadImage(l)
        label_a = sitk.GetArrayFromImage(label_nii)
        
        for label_idx in range(1, 16):
            label_mask = (label_a == label_idx).astype(np.float32)
            if label_mask.sum() > 0:
                print(f"  - Processing label {label_idx}/15", end='\r')
                resized = ndimage.zoom(label_mask, 
                    [total_shape[0]/label_mask.shape[0],
                     total_shape[1]/label_mask.shape[1],
                     total_shape[2]/label_mask.shape[2]], order=0)
                catlas[label_idx-1] += resized
                count[label_idx-1] += 1
        print()  # New line after processing all labels

    # Normalize and apply Gaussian smoothing
    print("\nStep 5: Normalizing and smoothing...")
    for i in range(15):
        if count[i] > 0:
            print(f"Processing organ {i+1}/15", end='\r')
            catlas[i] = catlas[i] / count[i]
            catlas[i] = gaussian_filter(catlas[i], sigma=3)
    print("\n✓ Normalization and smoothing complete")

    # Save the atlas
    print("\nStep 6: Saving atlas...")
    np.save("atlas_mm.npy", catlas)
    print("✓ Atlas saved successfully as atlas_mm.npy")
    print("\n=== Atlas Generation Complete ===")

if __name__ == "__main__":
    import time
    start_time = time.time()
    data_dir = "/Users/thurayaalzubaidi/multimodal-PL/amos/labelsTr"
    generate_atlas(data_dir)
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time/60:.2f} minutes")




