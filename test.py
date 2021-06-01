import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import disk, remove_small_objects
from tqdm import tqdm

from dataset.dataset_test import Test_Dataset
from model.UNet import UNet
import config


def _predict_single_image(model, dataloader, postprocess, prob_thresh,
        bone_thresh, size_thresh):
    pred = np.zeros(dataloader.dataset.image.shape)
    crop_size = dataloader.dataset.crop_size
    with torch.no_grad():
        for _, sample in enumerate(dataloader):
            images, centers = sample
            images = images
            output = model(images).sigmoid().cpu().numpy()

    return output


def predict(args):
    batch_size = 1
    postprocess = True if args.postprocess == "True" else False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet(1, args.n_labels)
    model.eval()
    if args.model_path is not None:
        model_weights = torch.load('best_model.pth')
        model.load_state_dict(model_weights['net'])

    image_path_list = sorted([os.path.join('.', file)
        for file in os.listdir('.') if "nii" in file])
    image_id_list = [os.path.basename(path).split("-")[0]
        for path in image_path_list]

    progress = tqdm(total=len(image_id_list))
    pred_info_list = []
    for image_id, image_path in zip(image_id_list, image_path_list):
        dataset = Test_Dataset(image_path, args)
        dataloader = DataLoader(dataset, batch_size, collate_fn=Test_Dataset.collate_fn)
        pred_image = _predict_single_image(model, dataloader, postprocess,
            args.prob_thresh, args.bone_thresh, args.size_thresh)
        pred_path = os.path.join('.', f"{image_id}_pred.nii.gz")
        nib.save(pred_image, pred_path)

        progress.update()


if __name__ == "__main__":
    args = config.args
    predict(args)
