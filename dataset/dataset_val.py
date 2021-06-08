import os
import sys
import random
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
from .transforms import Window, Normalize, Compose, RandomRotate, Center_Crop
import nibabel as nib
from skimage.measure import regionprops
from itertools import product

'''
class Val_Dataset(dataset):
    def __init__(self, args):

        self.args = args
        self.image_dir = os.path.join(args.dataset_path, "ribfrac-val-images")
        self.label_dir = os.path.join(args.dataset_path, "ribfrac-val-labels")

        self.files_prefix = sorted([x.split("-")[0]
            for x in os.listdir(self.image_dir)])
        self.transforms = Compose([
                Window(args.lower, args.upper),
                Normalize(args.lower, args.upper)
            ])
        self.train=True
        self.num_samples = 4
        self.crop_size = args.crop_size

    @staticmethod
    def _get_pos_centroids(label_arr):
        centroids = [tuple([round(x) for x in prop.centroid])
            for prop in regionprops(label_arr)]

        return centroids

    @staticmethod
    def _get_symmetric_neg_centroids(pos_centroids, x_size):
        sym_neg_centroids = [(x_size - x, y, z) for x, y, z in pos_centroids]

        return sym_neg_centroids

    @staticmethod
    def _get_spine_neg_centroids(shape, crop_size, num_samples):
        x_min, x_max = shape[0] // 2 - 40, shape[0] // 2 + 40
        y_min, y_max = 300, 400
        z_min, z_max = crop_size // 2, shape[2] - crop_size // 2
        spine_neg_centroids = [(
            np.random.randint(x_min, x_max),
            np.random.randint(y_min, y_max),
            np.random.randint(z_min, z_max)
        ) for _ in range(num_samples)]

        return spine_neg_centroids

    def _get_neg_centroids(self, pos_centroids, image_shape):
        num_pos = len(pos_centroids)
        sym_neg_centroids = self._get_symmetric_neg_centroids(
            pos_centroids, image_shape[0])

        if num_pos < self.num_samples // 2:
            spine_neg_centroids = self._get_spine_neg_centroids(image_shape,
                self.crop_size, self.num_samples - 2 * num_pos)
        else:
            spine_neg_centroids = self._get_spine_neg_centroids(image_shape,
                self.crop_size, num_pos)

        return sym_neg_centroids + spine_neg_centroids

    def _get_roi_centroids(self, label_arr):
        if self.train:
            # generate positive samples' centroids
            pos_centroids = self._get_pos_centroids(label_arr)

            # generate negative samples' centroids
            neg_centroids = self._get_neg_centroids(pos_centroids,
                label_arr.shape)

            # sample positives and negatives when necessary
            num_pos = len(pos_centroids)
            num_neg = len(neg_centroids)
            if num_pos >= self.num_samples:
                num_pos = self.num_samples // 2
                num_neg = self.num_samples // 2
            elif num_pos >= self.num_samples // 2:
                num_neg = self.num_samples - num_pos

            if num_pos < len(pos_centroids):
                pos_centroids = [pos_centroids[i] for i in np.random.choice(
                    range(0, len(pos_centroids)), size=num_pos, replace=False)]
            if num_neg < len(neg_centroids):
                neg_centroids = [neg_centroids[i] for i in np.random.choice(
                    range(0, len(neg_centroids)), size=num_neg, replace=False)]

            roi_centroids = pos_centroids + neg_centroids
        else:
            roi_centroids = [list(range(0, x, y // 2))[1:-1] + [x - y // 2]
                for x, y in zip(label_arr.shape, self.crop_size)]
            roi_centroids = list(product(*roi_centroids))

        roi_centroids = [tuple([int(x) for x in centroid])
            for centroid in roi_centroids]

        return roi_centroids

    def _crop_roi(self, arr, centroid):
        roi = np.ones(tuple([self.crop_size] * 3)) * (-1024)

        src_beg = [max(0, centroid[i] - self.crop_size // 2)
            for i in range(len(centroid))]
        src_end = [min(arr.shape[i], centroid[i] + self.crop_size // 2)
            for i in range(len(centroid))]
        dst_beg = [max(0, self.crop_size // 2 - centroid[i])
            for i in range(len(centroid))]
        dst_end = [min(arr.shape[i] - (centroid[i] - self.crop_size // 2),
            self.crop_size) for i in range(len(centroid))]
        roi[
            dst_beg[0]:dst_end[0],
            dst_beg[1]:dst_end[1],
            dst_beg[2]:dst_end[2],
        ] = arr[
            src_beg[0]:src_end[0],
            src_beg[1]:src_end[1],
            src_beg[2]:src_end[2],
        ]

        return roi

    def __getitem__(self, index):
        file_prefix = self.files_prefix[index]
        # read image and label
        img = sitk.ReadImage(os.path.join(self.image_dir, f"{file_prefix}-image.nii.gz"), sitk.sitkInt16)
        label = sitk.ReadImage(os.path.join(self.label_dir, f"{file_prefix}-label.nii.gz"), sitk.sitkUInt8)

        img_array = sitk.GetArrayFromImage(img)
        label_arr = sitk.GetArrayFromImage(label)

        image_arr = img_array.astype(np.float32)

        # calculate rois' centroids
        roi_centroids = self._get_roi_centroids(label_arr)

        # crop rois
        image_rois = [self._crop_roi(image_arr, centroid)
            for centroid in roi_centroids]
        label_rois = [self._crop_roi(label_arr, centroid)
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

    @staticmethod
    def collate_fn(samples):
        image_rois = torch.cat([x[0] for x in samples])
        label_rois = torch.cat([x[1] for x in samples])

        return image_rois, label_rois
'''

class Val_Dataset(dataset):
    def __init__(self, args):

        self.args = args
        self.filename_list = self.load_file_name_list(os.path.join(args.dataset_path, 'val_path_list.txt'))

        self.transforms = Compose([Center_Crop(base=16, max_size=args.val_crop_max_size)]) 

    def __getitem__(self, index):

        ct = sitk.ReadImage(self.filename_list[index][0], sitk.sitkInt16)
        seg = sitk.ReadImage(self.filename_list[index][1], sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        ct_array = ct_array / self.args.norm_factor
        ct_array = ct_array.astype(np.float32)

        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array).unsqueeze(0)

        if self.transforms:
            ct_array, seg_array = self.transforms(ct_array, seg_array)

        return ct_array, seg_array.squeeze(0)

    def __len__(self):
        return len(self.filename_list)

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                file_name_list.append(lines.split())
        return file_name_list