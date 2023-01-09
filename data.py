import os
import numpy as np
import tensorflow as tf
import h5py
import random
from functools import reduce

def get_item(path, rgb_transforms=None, depth_transforms=None):
    # TODO: Image transformations`
    with h5py.File(path, 'r') as f:
        rgb1, rgb2 = f['rgb1'], f['rgb2']
        sparse1, sparse2 = f['sparse1'], f['sparse2']
        dense1, dense2 = f['dense1'], f['dense2']
        input = tf.stack(
            [
                rgb1[0, :, :], rgb1[1, :, :], rgb1[2, :, :],
                rgb2[0, :, :], rgb2[1, :, :], rgb2[2, :, :],
                sparse1, sparse2
            ],
            axis=-1
        )
        target = tf.stack([dense1, dense2], axis=-1) # channels last
    return input, target

def get_files(root_dir, split):
    base_dir = os.path.join(root_dir, split)
    dirs = [os.path.join(base_dir, dir) for dir in os.listdir(base_dir)]
    files_list = [[os.path.join(dir, file) for file in os.listdir(dir)] for dir in dirs]
    files = reduce(lambda a, b: a + b, files_list)
    return files

def dataset_generator(root_dir, split, shuffle):
    files = get_files(root_dir, split)
    if shuffle:
        random.shuffle(files)

    idx = 0
    while idx < len(files):
        yield get_item(files[idx])
        idx += 1

def get_dataset_size(root_dir, split):
    files = get_files(root_dir, split)
    return len(files)
