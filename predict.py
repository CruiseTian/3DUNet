import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import disk, remove_small_objects
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset.test_dataset import TestDataset
from dataset import transforms as tsfm
from model.UNet import UNet
import config
from utils.common import test_collate_fn


def post_process(pred, image, prob_thresh, bone_thresh, size_thresh):

    # 去除低置信度预测
    pred = np.where(pred > prob_thresh, pred, 0)

    # 去除误报
    image_bone = image > bone_thresh
    image_bone_2d = image_bone.sum(axis=-1)
    image_bone_2d = ndimage.median_filter(image_bone_2d, 10)
    image_spine = (image_bone_2d > image_bone_2d.max() // 3)
    kernel = disk(7)
    image_spine = ndimage.binary_opening(image_spine, kernel)
    image_spine = ndimage.binary_closing(image_spine, kernel)
    image_spine_label = label(image_spine)
    max_area = 0

    for region in regionprops(image_spine_label):
        if region.area > max_area:
            max_region = region
            max_area = max_region.area
    image_spine = np.zeros_like(image_spine)
    image_spine[
        max_region.bbox[0]:max_region.bbox[2],
        max_region.bbox[1]:max_region.bbox[3]
    ] = max_region.convex_image > 0
    pred = np.where(image_spine[..., np.newaxis], 0, pred)

    # 移除小连接区域
    pred_bin = pred > 0
    pred_bin = remove_small_objects(pred_bin, size_thresh)
    pred = np.where(pred_bin, pred, 0)

    return pred


def predict_single_image(model, dataloader, postprocess, prob_thresh,
        bone_thresh, size_thresh):
    pred = np.zeros(dataloader.dataset.image.shape)
    crop_size = dataloader.dataset.crop_size
    with torch.no_grad():
        for _, sample in enumerate(dataloader):
            images, centers = sample
            images = images.cuda()
            output = model(images).sigmoid().cpu().numpy()
            for i in range(len(centers)):
                center_x, center_y, center_z = centers[i]
                cur_pred_patch = pred[
                    center_x - crop_size // 2:center_x + crop_size // 2,
                    center_y - crop_size // 2:center_y + crop_size // 2,
                    center_z - crop_size // 2:center_z + crop_size // 2
                ]
                pred[
                    center_x - crop_size // 2:center_x + crop_size // 2,
                    center_y - crop_size // 2:center_y + crop_size // 2,
                    center_z - crop_size // 2:center_z + crop_size // 2
                ] = np.where(cur_pred_patch > 0, np.mean((output[i,1],
                    cur_pred_patch), axis=0), output[i,1])


    if postprocess:
        pred = post_process(pred, dataloader.dataset.image, prob_thresh,
            bone_thresh, size_thresh)

    return pred


def make_submission_files(pred, image_id, affine):
    pred_label = label(pred > 0).astype(np.int16)
    pred_regions = regionprops(pred_label, pred)
    pred_index = [0] + [region.label for region in pred_regions]
    pred_proba = [0.0] + [region.mean_intensity for region in pred_regions]
    
    pred_label_code = [0] + [1] * int(pred_label.max())
    pred_image = nib.Nifti1Image(pred_label, affine)
    pred_info = pd.DataFrame({
        "public_id": [image_id] * len(pred_index),
        "label_id": pred_index,
        "confidence": pred_proba,
        "label_code": pred_label_code
    })

    return pred_image, pred_info


def predict(args):
    if not os.path.exists(args.pred_dir): os.makedirs(args.pred_dir)
    num_workers = 0
    batch_size = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet(1, 2).to(device)
    model = torch.nn.DataParallel(model, device_ids=[args.gpu])
    model.eval()
    if args.model_path is not None:
        model_weights = torch.load(args.model_path)
        model.load_state_dict(model_weights['net'])

    image_path_list = sorted([os.path.join(args.test_data_path, file)
        for file in os.listdir(args.test_data_path) if "nii" in file])
    image_id_list = [os.path.basename(path).split("-")[0]
        for path in image_path_list]

    progress = tqdm(total=len(image_id_list))
    pred_info_list = []
    for image_id, image_path in zip(image_id_list, image_path_list):
        dataset = TestDataset(image_path, args)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=test_collate_fn)
        pred_arr = predict_single_image(model, dataloader, args.postprocess,
            args.prob_thresh, args.bone_thresh, args.size_thresh)
        pred_image, pred_info = make_submission_files(pred_arr, image_id,
            dataset.image_affine)
        pred_info_list.append(pred_info)
        pred_path = os.path.join(args.pred_dir, f"{image_id}-label.nii.gz")
        nib.save(pred_image, pred_path)

        progress.update()

    pred_info = pd.concat(pred_info_list, ignore_index=True)
    pred_info.to_csv(os.path.join(args.pred_dir, "pred_info.csv"),
        index=False)


if __name__ == "__main__":
    args = config.args
    predict(args)
