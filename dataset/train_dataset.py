import os
import sys
import random
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset, DataLoader
from .transforms import Window, Normalize, Compose
from utils.process import get_roi_centroids, crop_roi

class TrainDataset(dataset):
    def __init__(self, args, image_dir, label_dir=None, train=True):

        self.args = args
        self.image_dir = image_dir
        self.label_dir = label_dir

        self.files_prefix = sorted([x.split("-")[0]
            for x in os.listdir(self.image_dir)])
        self.transforms = Compose([
                Window(args.lower, args.upper),
                Normalize(args.lower, args.upper)
            ])
        self.train=train
        self.num_samples = 4
        self.crop_size = 64

    def __getitem__(self, index):
        file_prefix = self.files_prefix[index]
        # read image and label
        img = sitk.ReadImage(os.path.join(self.image_dir, f"{file_prefix}-image.nii.gz"), sitk.sitkInt16)
        label = sitk.ReadImage(os.path.join(self.label_dir, f"{file_prefix}-label.nii.gz"), sitk.sitkUInt8)

        img_array = sitk.GetArrayFromImage(img)
        label_arr = sitk.GetArrayFromImage(label)

        image_arr = img_array.astype(np.float32)

        roi_centroids = get_roi_centroids(label_arr, self.crop_size, self.num_samples, self.train)

        image_rois = [crop_roi(image_arr, centroid, self.crop_size)
            for centroid in roi_centroids]
        label_rois = [crop_roi(label_arr, centroid, self.crop_size)
            for centroid in roi_centroids]

        if self.transforms:
            image_rois = self.transforms(image_rois)

        image_rois = torch.tensor(np.stack(image_rois)[:, np.newaxis],
            dtype=torch.float)
        label_rois = (np.stack(label_rois) > 0).astype(np.float)
        label_rois = torch.tensor(label_rois[:, np.newaxis],
            dtype=torch.float)

        return image_rois, label_rois

    def __len__(self):
        return len(self.files_prefix)