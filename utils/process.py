import numpy as np
import torch
from skimage.measure import regionprops
from itertools import product

def get_pos_centroids(label_arr):
    centroids = [tuple([round(x) for x in prop.centroid])
        for prop in regionprops(label_arr)]

    return centroids

def get_symmetric_neg_centroids(pos_centroids, x_size):
    sym_neg_centroids = [(x_size - x, y, z) for x, y, z in pos_centroids]

    return sym_neg_centroids

def get_spine_neg_centroids(shape, crop_size, num_samples):
    x_min, x_max = shape[0] // 2 - 40, shape[0] // 2 + 40
    y_min, y_max = 300, 400
    z_min, z_max = crop_size // 2, shape[2] - crop_size // 2
    spine_neg_centroids = [(
        np.random.randint(x_min, x_max),
        np.random.randint(y_min, y_max),
        np.random.randint(z_min, z_max)
    ) for _ in range(num_samples)]

    return spine_neg_centroids

def get_neg_centroids(pos_centroids, image_shape, crop_size=64, num_samples=4):
    num_pos = len(pos_centroids)
    sym_neg_centroids = get_symmetric_neg_centroids(
        pos_centroids, image_shape[0])

    if num_pos < num_samples // 2:
        spine_neg_centroids = get_spine_neg_centroids(image_shape,
            crop_size, num_samples - 2 * num_pos)
    else:
        spine_neg_centroids = get_spine_neg_centroids(image_shape,
            crop_size, num_pos)

    return sym_neg_centroids + spine_neg_centroids

def get_roi_centroids(label_arr, crop_size=64, num_samples=4, train=True):
    if train:
        # generate positive samples' centroids
        pos_centroids = get_pos_centroids(label_arr)

        # generate negative samples' centroids
        neg_centroids = get_neg_centroids(pos_centroids, label_arr.shape, crop_size, num_samples)

        # sample positives and negatives when necessary
        num_pos = len(pos_centroids)
        num_neg = len(neg_centroids)
        if num_pos >= num_samples:
            num_pos = num_samples // 2
            num_neg = num_samples // 2
        elif num_pos >= num_samples // 2:
            num_neg = num_samples - num_pos

        if num_pos < len(pos_centroids):
            pos_centroids = [pos_centroids[i] for i in np.random.choice(
                range(0, len(pos_centroids)), size=num_pos, replace=False)]
        if num_neg < len(neg_centroids):
            neg_centroids = [neg_centroids[i] for i in np.random.choice(
                range(0, len(neg_centroids)), size=num_neg, replace=False)]

        roi_centroids = pos_centroids + neg_centroids
    else:
        roi_centroids = [list(range(0, x, y // 2))[1:-1] + [x - y // 2]
            for x, y in zip(label_arr.shape, crop_size)]
        roi_centroids = list(product(*roi_centroids))

    roi_centroids = [tuple([int(x) for x in centroid])
        for centroid in roi_centroids]

    return roi_centroids

def crop_roi(arr, centroid, crop_size=64):
    roi = np.ones(tuple([crop_size] * 3)) * (-1024)

    src_beg = [max(0, centroid[i] - crop_size // 2)
        for i in range(len(centroid))]
    src_end = [min(arr.shape[i], centroid[i] + crop_size // 2)
        for i in range(len(centroid))]
    dst_beg = [max(0, crop_size // 2 - centroid[i])
        for i in range(len(centroid))]
    dst_end = [min(arr.shape[i] - (centroid[i] - crop_size // 2),
        crop_size) for i in range(len(centroid))]
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