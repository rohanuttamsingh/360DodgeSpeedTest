#!/usr/bin/python3
import argparse
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models import generate_s2d_model, generate_smaller_model, generate_even_smaller_model
from losses import L1Loss
from data import get_item
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help='Options: s2d | smaller')
parser.add_argument('-c', '--checkpoint_path', type=str, help='Path to model checkpoint')
parser.add_argument('-i', '--input_path', type=str, help='Path to input file to test on')
parser.add_argument('-n', '--normalized', action='store_true', help='Set if the dataset is normalized')
args = parser.parse_args()

if args.model == 's2d':
    model = generate_s2d_model(args.normalized)
elif args.model == 'smaller':
    model = generate_smaller_model(args.normalized)
elif args.model == 'even':
    model = generate_even_smaller_model(args.normalized)
model.load_weights(args.checkpoint_path)

loss_object = L1Loss()

input, target = get_item(args.input_path, args.normalized, False)
input = tf.expand_dims(input, 0)
target = tf.expand_dims(target, 0)

start_time = datetime.now()
pred = model(input, training=False)
end_time = datetime.now()
seconds = (end_time - start_time).total_seconds()
hertz = 1 / seconds
print(f'{seconds} s')
print(f'{hertz} Hz')

loss = loss_object(target, pred)
print(f'Loss: {loss}')

input_ = np.array(input).squeeze()
target_ = np.array(target).squeeze()
pred_ = np.array(pred).squeeze()

rgb1 = input_[:, :, 0:3]
rgb2 = input_[:, :, 3:6]
if not args.normalized:
    sparse1 = input_[:, :, 6]
    sparse2 = input_[:, :, 7]
dense1 = target_[:, :, 0]
dense2 = target_[:, :, 1]
pred1 = pred_[:, :, 0]
pred2 = pred_[:, :, 1]

vmax = 1 if args.normalized else 4
# Max depth is 1 (no units) if normalized, otherwise 4m by L5 sensor limits
if args.normalized:
    fig, ax = plt.subplots(2, 3)
    ax[0][0].imshow(rgb1)
    ax[0][1].imshow(dense1, cmap='gray_r', vmin=0, vmax=vmax)
    ax[0][2].imshow(pred1, cmap='gray_r', vmin=0, vmax=vmax)
    ax[1][0].imshow(rgb2)
    ax[1][1].imshow(dense2, cmap='gray_r', vmin=0, vmax=vmax)
    ax[1][2].imshow(pred2, cmap='gray_r', vmin=0, vmax=vmax)
else:
    fig, ax = plt.subplots(2, 4)
    ax[0][0].imshow(rgb1)
    ax[0][1].imshow(sparse1, cmap='gray_r', vmin=0, vmax=vmax)
    ax[0][2].imshow(dense1, cmap='gray_r', vmin=0, vmax=vmax)
    ax[0][3].imshow(pred1, cmap='gray_r', vmin=0, vmax=vmax)
    ax[1][0].imshow(rgb2)
    ax[1][1].imshow(sparse2, cmap='gray_r', vmin=0, vmax=vmax)
    ax[1][2].imshow(dense2, cmap='gray_r', vmin=0, vmax=vmax)
    ax[1][3].imshow(pred2, cmap='gray_r', vmin=0, vmax=vmax)
plt.show()
