#!/usr/bin/python3
import argparse
import os
import h5py
import numpy as np

def _normalize_depth(depth):
    return (depth - np.amin(depth)) / (np.amax(depth) - np.amin(depth))

def normalize_depth(path):
    with h5py.File(path, 'r') as f:
        rgb1, rgb2 = np.array(f['rgb1']), np.array(f['rgb2'])
        depth1, depth2 = np.array(f['depth1']), np.array(f['depth2'])
    depth1, depth2 = _normalize_depth(depth1), _normalize_depth(depth2)
    with h5py.File(path, 'w') as f:
        f.create_dataset('rgb1', data=rgb1)
        f.create_dataset('rgb2', data=rgb2)
        f.create_dataset('dense1', data=depth1)
        f.create_dataset('dense2', data=depth2)

def normalize_dir(path):
    print(f'Started normalizing directory {path}')
    for filename in os.listdir(path):
        normalize_depth(os.path.join(path, filename))
    print(f'Finished normalizing directory {path}')

def normalize_all_dirs(base_path):
    train_path = os.path.join(base_path, 'train')
    val_path = os.path.join(base_path, 'val')
    for dir_ in os.listdir(train_path):
        normalize_dir(os.path.join(train_path, dir_))
    for dir_ in os.listdir(val_path):
        normalize_dir(os.path.join(val_path, dir_))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Normalize depth of dataset')
    parser.add_argument('--path', type=str, help='Path to base directory')
    args = parser.parse_args()

    normalize_all_dirs(args.path)
