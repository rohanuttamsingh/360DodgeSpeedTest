#!/usr/bin/python3
import numpy as np
import os
import h5py
import argparse
from constants import ROWS, COLS

def dense_to_l5(dense):
    l5 = np.zeros_like(dense)
    for start_row in range(0, ROWS, ROWS // 8):
        end_row = start_row + ROWS // 8
        for start_col in range(0, COLS, COLS // 8):
            end_col = start_col + COLS // 8
            median = np.median(dense[start_row : end_row, start_col : end_col])
            l5[start_row : end_row, start_col : end_col] = median
    return l5

def threshold_depth(depth, threshold):
    if threshold > 0:
        return np.minimum(depth, threshold)
    return depth

def convert_to_l5(path, threshold):
    with h5py.File(path, 'r') as f:
        rgb1, rgb2 = np.array(f['rgb1']), np.array(f['rgb2'])
        dense1, dense2 = np.array(f['depth1']), np.array(f['depth2'])
    dense1, dense2 = threshold_depth(dense1, threshold), threshold_depth(dense2, threshold)
    # sparse1, sparse2 = dense_to_l5(dense1), dense_to_l5(dense2)
    with h5py.File(path, 'w') as f:
        f.create_dataset('rgb1', data=rgb1)
        f.create_dataset('rgb2', data=rgb2)
        f.create_dataset('dense1', data=dense1)
        f.create_dataset('dense2', data=dense2)
        # f.create_dataset('sparse1', data=sparse1)
        # f.create_dataset('sparse2', data=sparse2)

def convert_dir_to_l5(path, threshold):
    print(f'Started converting directory {path}')
    for filename in os.listdir(path):
        convert_to_l5(os.path.join(path, filename), threshold)
    print(f'Finished converting directory {path}')

def convert_all_dirs(base_path, threshold):
    train_path = os.path.join(base_path, 'train')
    val_path = os.path.join(base_path, 'val')
    for dir_ in os.listdir(train_path):
        convert_dir_to_l5(os.path.join(train_path, dir_), threshold)
    for dir_ in os.listdir(val_path):
        convert_dir_to_l5(os.path.join(val_path, dir_), threshold)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert mono to stereo images')
    parser.add_argument('--path', type=str, help='Path to directory')
    parser.add_argument('--threshold', type=int, help='Depth threshold')
    args = parser.parse_args()

    convert_all_dirs(args.path, args.threshold)
