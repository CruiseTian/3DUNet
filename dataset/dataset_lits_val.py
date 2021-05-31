from posixpath import join
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
from .transforms import Center_Crop, Compose


class Val_Dataset(dataset):
    def __init__(self, args):

        self.args = args
        self.image_dir = os.path.join(args.dataset_path, "ribfrac-val-images")
        self.label_dir = os.path.join(args.dataset_path, "ribfrac-val-labels")

        self.files_prefix = sorted([x.split("-")[0]
            for x in os.listdir(self.image_dir)])

        self.transforms = Compose([Center_Crop(base=16, max_size=args.val_crop_max_size)]) 

    def __getitem__(self, index):
        file_prefix = self.files_prefix[index]
        img = sitk.ReadImage(os.path.join(self.image_dir, f"{file_prefix}-image.nii.gz"), sitk.sitkInt16)
        label = sitk.ReadImage(os.path.join(self.image_dir, f"{file_prefix}-label.nii.gz"), sitk.sitkUInt8)

        img_array = sitk.GetArrayFromImage(img)
        label_array = sitk.GetArrayFromImage(label)

        img_array = img_array.astype(np.float32)

        img_array = torch.FloatTensor(img_array).unsqueeze(0)
        label_array = torch.FloatTensor(label_array).unsqueeze(0)

        if self.transforms:
            img_array, label_array = self.transforms(img_array, label_array)     

        return img_array, label_array.squeeze(0)

    def __len__(self):
        return len(self.files_prefix)