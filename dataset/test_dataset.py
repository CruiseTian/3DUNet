from posixpath import join
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
import nibabel as nib
from itertools import product
import torch
from torch.utils.data import Dataset as dataset
from .transforms import Window, Normalize, Compose


class TestDataset(dataset):
    def __init__(self, img_path, args):

        self.args = args
        image = nib.load(img_path)
        self.image_affine = image.affine
        self.image = image.get_fdata().astype(np.int16)
        
        self.crop_size = args.crop_size
        self.transforms = Compose([
                Window(args.lower, args.upper),
                Normalize(args.lower, args.upper)
            ])
        self.centers = self._get_centers()
    
    def _get_centers(self):
        dim_coords = [list(range(0, dim, self.crop_size // 2))[1:-1]\
            + [dim - self.crop_size // 2] for dim in self.image.shape]
        centers = list(product(*dim_coords))

        return centers

    def __len__(self):
        return len(self.centers)

    def _crop_patch(self, idx):
        center_x, center_y, center_z = self.centers[idx]
        patch = self.image[
            center_x - self.crop_size // 2:center_x + self.crop_size // 2,
            center_y - self.crop_size // 2:center_y + self.crop_size // 2,
            center_z - self.crop_size // 2:center_z + self.crop_size // 2
        ]

        return patch

    def __getitem__(self, idx):
        image = self._crop_patch(idx)
        center = self.centers[idx]

        if self.transforms:
            image = self.transforms(image)

        image = torch.tensor(image[np.newaxis], dtype=torch.float)

        return image, center