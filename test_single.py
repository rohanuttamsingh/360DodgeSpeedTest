#!/usr/bin/python3
import argparse
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models import generate_model
from losses import L1Loss
from data import get_item
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('-checkpoint_path', type=str, help='Path to model checkpoint')
parser.add_argument('-input_path', type=str, help='Path to input file to test on')
args = parser.parse_args()

model = generate_model()
model.load_weights(args.checkpoint_path)

loss_object = L1Loss()

input, target = get_item(args.input_path)
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
sparse1 = input_[:, :, 6]
sparse2 = input_[:, :, 7]
dense1 = target_[:, :, 0]
dense2 = target_[:, :, 1]
pred1 = pred_[:, :, 0]
pred2 = pred_[:, :, 1]

fig, ax = plt.subplots(2, 4)
ax[0][0].imshow(rgb1)
ax[0][1].imshow(sparse1)
ax[0][2].imshow(dense1)
ax[0][3].imshow(pred1)
ax[1][0].imshow(rgb2)
ax[1][1].imshow(sparse2)
ax[1][2].imshow(dense2)
ax[1][3].imshow(pred2)
plt.show()
