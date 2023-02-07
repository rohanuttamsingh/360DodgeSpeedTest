import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import h5py
import random
from functools import reduce
from constants import L5_THRESHOLD, ROWS, COLS
from l5_block import threshold_depth, dense_to_l5
from normalize import _normalize_depth as normalize_depth

def rotate(rgb1, rgb2, dense1, dense2):
    # [-5, 5] from Sparse-to-Dense
    rads = 5 * np.pi / 180
    degree = random.uniform(-rads, rads)
    rgb1 = tfa.image.rotate(rgb1, degree, interpolation='bilinear', fill_mode='nearest')
    rgb2 = tfa.image.rotate(rgb2, degree, interpolation='bilinear', fill_mode='nearest')
    dense1 = tfa.image.rotate(dense1, degree, interpolation='bilinear', fill_mode='nearest')
    dense2 = tfa.image.rotate(dense2, degree, interpolation='bilinear', fill_mode='nearest')
    return rgb1, rgb2, dense1, dense2

def brightness(rgb1, rgb2):
    # 0.6-1.4 from Sparse-to-Dense appears to translate to -0.6-0.4
    factor = random.uniform(-0.6, 0.4)
    rgb1 = tf.image.adjust_brightness(rgb1, factor)
    rgb2 = tf.image.adjust_brightness(rgb2, factor)
    return rgb1, rgb2

def contrast(rgb1, rgb2):
    # 0.6-1.4 from Sparse-to-Dense
    factor = random.uniform(0.6, 1.4)
    rgb1 = tf.image.adjust_contrast(rgb1, factor)
    rgb2 = tf.image.adjust_contrast(rgb2, factor)
    return rgb1, rgb2

def saturate(rgb1, rgb2):
    # 0.6-1.4 from Sparse-To-Dense
    factor = random.uniform(0.6, 1.4)
    rgb1 = tf.image.adjust_saturation(rgb1, factor)
    rgb2 = tf.image.adjust_saturation(rgb2, factor)
    return rgb1, rgb2

def flip(rgb1, rgb2, dense1, dense2):
    # 50% probability from Sparse-to-Dense
    if random.random() > 0.5:
        rgb1 = tf.image.flip_left_right(rgb1)
        rgb2 = tf.image.flip_left_right(rgb2)
        dense1 = tf.image.flip_left_right(dense1)
        dense2 = tf.image.flip_left_right(dense2)
    return rgb1, rgb2, dense1, dense2

def scale(rgb1, rgb2, dense1, dense2):
    # [1, 1.5] from Sparse-to-Dense
    factor = random.uniform(1, 1.5)
    central_fraction = 1 / factor
    rgb1 = tf.cast(tf.image.resize(tf.image.central_crop(rgb1, central_fraction), (ROWS, COLS), method='bilinear'), tf.int32)
    rgb2 = tf.cast(tf.image.resize(tf.image.central_crop(rgb2, central_fraction), (ROWS, COLS), method='bilinear'), tf.int32)
    dense1 = tf.image.resize(tf.image.central_crop(dense1, central_fraction) / factor, (ROWS, COLS), method='bilinear')
    dense2 = tf.image.resize(tf.image.central_crop(dense2, central_fraction) / factor, (ROWS, COLS), method='bilinear')
    return rgb1, rgb2, dense1, dense2

def random_augment(rgb1, rgb2, dense1, dense2):
    # rgb1, rgb2, dense1, dense2 = rotate(rgb1, rgb2, dense1, dense2)
    rgb1, rgb2 = brightness(rgb1, rgb2)
    rgb1, rgb2 = contrast(rgb1, rgb2)
    rgb1, rgb2 = saturate(rgb1, rgb2)
    dense1, dense2 = np.expand_dims(dense1, axis=-1), np.expand_dims(dense2, axis=-1)
    rgb1, rgb2, dense1, dense2 = flip(rgb1, rgb2, dense1, dense2)
    rgb1, rgb2, dense1, dense2 = scale(rgb1, rgb2, dense1, dense2)
    dense1, dense2 = np.squeeze(dense1), np.squeeze(dense2)
    return rgb1, rgb2, dense1, dense2

def get_item(path, normalized, augment):
    with h5py.File(path, 'r') as f:
        rgb1, rgb2 = np.array(f['rgb1']), np.array(f['rgb2'])
        dense1, dense2 = np.array(f['dense1']), np.array(f['dense2'])
        rgb1 = np.swapaxes(rgb1, 0, 1)
        rgb1 = np.swapaxes(rgb1, 1, 2)
        rgb2 = np.swapaxes(rgb2, 0, 1)
        rgb2 = np.swapaxes(rgb2, 1, 2)
        if augment:
            rgb1, rgb2, dense1, dense2 = random_augment(rgb1, rgb2, dense1, dense2)
        if normalized:
            dense1, dense2 = normalize_depth(dense1), normalize_depth(dense2)
            input = tf.stack(
                [
                    rgb1[:, :, 0], rgb1[:, :, 1], rgb1[:, :, 2],
                    rgb2[:, :, 0], rgb2[:, :, 1], rgb2[:, :, 2]
                ],
                axis=-1
            )
        else:
            dense1, dense2 = threshold_depth(dense1, L5_THRESHOLD), threshold_depth(dense2, L5_THRESHOLD)
            sparse1, sparse2 = dense_to_l5(dense1), dense_to_l5(dense2)
            input = tf.stack(
                [
                    rgb1[:, :, 0], rgb1[:, :, 1], rgb1[:, :, 2],
                    rgb2[:, :, 0], rgb2[:, :, 1], rgb2[:, :, 2],
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

def dataset_generator(root_dir, split, shuffle, normalized, augment):
    files = get_files(root_dir, split)
    if shuffle:
        random.shuffle(files)

    idx = 0
    while idx < len(files):
        yield get_item(files[idx], normalized, augment)
        idx += 1

def get_dataset_size(root_dir, split):
    files = get_files(root_dir, split)
    return len(files)

def get_dataset_in_memory(root_dir, split, shuffle, normalized, augment):
    files = get_files(root_dir, split)
    if shuffle:
        random.shuffle(files)
    items = [get_item(file, normalized, augment) for file in files]
    inputs = [item[0] for item in items]
    targets = [item[1] for item in items]
    ds = tf.data.Dataset.from_tensor_slices((inputs, targets))
    return ds

def get_inputs_in_memory(root_dir, split, shuffle, normalized, augment):
    files = get_files(root_dir, split)
    if shuffle:
        random.shuffle(files)
    inputs = [get_item(file, normalized, augment)[0] for file in files]
    return inputs
