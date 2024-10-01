import numpy as np
import os
import SimpleITK as sitk
import glob
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage as ndimage
from .transforms import transform_preprocessing_amos

import yaml
import SimpleITK as sitk

PATH_TO_CONFIG = Path("./config/")


def get_config(config_name):
    """Loads a .yaml file from ./config corresponding to the name arg.

    Args:
        config_name: A string referring to the .yaml file to load.

    Returns:
        A container including the information of the referred .yaml file and information
        regarding the dataset, if specified in the referred .yaml file.
    """
    with open(PATH_TO_CONFIG / (config_name + '.yaml'), 'r') as stream:
        config = yaml.safe_load(stream)

    # Add dataset config
    if 'dataset' in config:
        path_to_data_info = Path(os.getcwd()) / 'dataset' / config['dataset'] / 'data_info.json'
        config.update(load_json(path_to_data_info))

    return config

def getmaxcomponent(mask_array, num_limit=10, min_filter = 1e6):
    # sitk方法获取连通域
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    cca.FullyConnectedOff()
    _input = sitk.GetImageFromArray(mask_array.astype(np.uint8))
    output_ex = cca.Execute(_input)
    labeled_img = sitk.GetArrayFromImage(output_ex)
    num = cca.GetObjectCount()
    max_label = 0
    max_num = 0
    # 不必遍历全部连通域，一般在前面就有对应全身mask的label，减少计算时间
    for i in range(1, num_limit):  	
        if np.sum(labeled_img == i) < min_filter:		# 全身mask的体素数量必然很大，小于设定值的不考虑
            continue
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    if max_label == 0:
        return None
    maxcomponent = np.array((labeled_img == max_label)).astype(np.uint8)
    print(str(max_label) + ' num:' + str(max_num))  	# 看第几个是最大的
    return maxcomponent.astype(np.uint8)


def get_body(CT_nii_array, threshold_all = -200, min_filter = 1e6):
    """
    卡CT阈值获取身体（理想情况下够用了，不过多数情况下会包括到机床部分）
    """
    # 阈值二值化，获得最大的3d的连通域
    CT_array = np.zeros_like(CT_nii_array)
    # threshold_all# 卡的阈值，卡出整个身体以及机床部分
    CT_array[CT_nii_array >= threshold_all] = 1
    CT_array[CT_nii_array <= threshold_all] = 0

    CT_array = ndimage.binary_erosion(CT_array, structure=np.ones((2,2,2)))

    #CT_array = ndimage.binary_erosion(CT_array, structure=np.ones((3,3,3)))
    maxcomponent = getmaxcomponent(CT_array, 100, min_filter=min_filter)
    if maxcomponent is None:
        maxcomponent = (CT_nii_array > thre).astype(np.float32)
        print("Get max component failed. ", maxcomponent.sum())
        maxcomponent = ndimage.binary_erosion(maxcomponent, structure=np.ones((10,10,10)))
        maxcomponent = ndimage.binary_dilation(maxcomponent, structure=np.ones((10,10,10)))

    return maxcomponent

preprocessing_config = get_config('preprocessing_amos')
_preprocessing_transform = transform_preprocessing_amos(
            margin=preprocessing_config['margin'],
            crop_key=preprocessing_config['key'], 
            orientation=preprocessing_config['orientation'],
            resize_shape=preprocessing_config['resize_shape']
        )

datapath = "/apdcephfs/share_1290796/lh/dataset/nnFormer_raw/nnFormer_raw_data/Task005_AMOS/imagesTr"
files = glob.glob(os.path.join(datapath, "*.nii.gz"))

print(f"Totally {len(files)} files.")

for idx, l in enumerate(files):

    label_path = l.replace("imagesTr", "labelsTr").replace("_0000", "")

    cid = int(label_path.split("/")[-1].split("_")[1][:-7])

    #if cid > 500:
    #    continue

    if cid != 530:
        continue

    print(f"Now, process {idx}, {l}")

    CT_nii = sitk.ReadImage(l)
    CT_a = sitk.GetArrayFromImage(CT_nii)
    

    label_nii = sitk.ReadImage(label_path)
    image_r = sitk.GetArrayFromImage(sitk.ReadImage(l))
    label_a = sitk.GetArrayFromImage(label_nii)

    ##################################### monai process: orientation and spacing normlization 

    case_dict = {
                'image': l,
                'label': label_path
            }

    preprocessed_case = _preprocessing_transform(case_dict)
    image, label = preprocessed_case['image'].squeeze(), preprocessed_case['label'].squeeze()

    label[label >= 14] = 0
    orilabelsum = label.sum()
    print(f"ori size: {CT_a.shape} {label_a.shape}")
    print(f"pre size: {image.shape}, {label.shape}, {label.sum()}")


    ##################################### Crop nan information slices with label

    print("before image crop ", image.shape)
    z_indexes, y_indexes, x_indexes = np.nonzero(label != 0)
    zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
    zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]

    image = image[:, :, xmin:xmax]
    label = label[:, :, xmin:xmax]

    ##################################### Crop tissue areas

    if cid == 540 or cid == 518:  # some special cases
        thre = 30000

    elif cid > 410:
        thre = 25   # mri
    else:
        thre = -200   # CT

    maxcomponent = get_body(image, thre)
        
    z_indexes, y_indexes, x_indexes = np.nonzero(maxcomponent != 0)
    zmin, ymin, xmin = [max(0, int(np.min(arr) - 3)) for arr in (z_indexes, y_indexes, x_indexes)]
    zmax, ymax, xmax = [int(np.max(arr) + 3) for arr in (z_indexes, y_indexes, x_indexes)]

    image_a = image[zmin:zmax, ymin:ymax, xmin:xmax]
    label_a = label[zmin:zmax, ymin:ymax, xmin:xmax]

    print(f"mid_size: {image_a.shape}, {label.shape}, {maxcomponent.shape}, {label_a.sum()}")

    ################################### Crop out hand areas
    image_up = image[:, :, :image_a.shape[2] // 2 + 10] # for 532, this is 0

    maxcomponent_up = get_body(image_up, thre, min_filter=1e5)
    print("inter")

    z_indexes_up, y_indexes_up, x_indexes_up = np.nonzero(maxcomponent_up != 0)
    zmin_up, ymin_up, xmin_up = [max(0, int(np.min(arr) - 5)) for arr in (z_indexes_up, y_indexes_up, x_indexes_up)] # for 594, this is 12
    zmax_up, ymax_up, xmax_up = [int(np.max(arr) + 5) for arr in (z_indexes_up, y_indexes_up, x_indexes_up)]


    inter_up = zmax_up - zmin_up
    inter_all = zmax - zmin

    if inter_all - inter_up > 30 and cid > 500:
        print("find hand")
        image_a = image_a[zmin_up:zmax_up, :, :]
        label_a = label_a[zmin_up:zmax_up, :, :]
        maxcomponent = maxcomponent[zmin_up:zmax_up, :, :]
    else:
        print("xxx", inter_all, inter_up)

    print(f"aftersize: {image_a.shape}, {label.shape}, {maxcomponent.shape}, {label_a.sum()}")
    if label_a.sum() != orilabelsum:
        print("fxxxk")

    CT_nii_pred = sitk.GetImageFromArray(image_a)
    CT_nii_pred.SetSpacing((1,1,2))
    #CT_nii_pred.SetOrigin(CT_nii.GetOrigin())
    #CT_nii_pred.SetDirection(CT_nii.GetDirection())

    label_nii_pred = sitk.GetImageFromArray(label_a)
    label_nii_pred.SetSpacing((1,1,2))
    #label_nii_pred.SetOrigin(label_nii.GetOrigin())
    #label_nii_pred.SetDirection(label_nii.GetDirection())

    cpath_image = os.path.join("/apdcephfs/share_1290796/lh/transoar-main/preprocess/processed_data_f/imagesTr", l.split("/")[-1])
    cpath_label = os.path.join("/apdcephfs/share_1290796/lh/transoar-main/preprocess/processed_data_f/labelsTr", label_path.split("/")[-1])

    #sitk.WriteImage(CT_nii_pred, cpath_image)
    #sitk.WriteImage(label_nii_pred, cpath_label)
    #plt.subplot(2,1,2)

    plt.imshow(maxcomponent[zmin:zmax, ymin:ymax, xmin:xmax][:,:,image_a.shape[2] // 2], cmap = "gray")
    #plt.imshow(maxcomponent_up[zmin:zmax, ymin:ymax, xmin:xmax][:,:,20], cmap = "gray")
    # plt.imshow(image_a[:,:,image_a.shape[2] // 2], cmap = "gray")
    plt.savefig(os.path.join("/apdcephfs/share_1290796/lh/transoar-main/preprocess/processed_data_f/pics", l.split("/")[-1][:-7]+".png"))

    