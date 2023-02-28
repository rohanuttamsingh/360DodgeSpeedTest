#!/usr/bin/python3
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models import (
    generate_s2d_model,
    generate_smaller_model,
    generate_even_smaller_model,
    generate_even_smaller_upproj_model,
    generate_tiny_model,
    generate_tiny_resnet_deconv_model,
    generate_tiny_upproj_model,
    generate_tiny_half_upproj_model,
    generate_tiny_fast_upproj_model,
    generate_tiny_faster_upproj_model,
    generate_teeny_fast_upproj_model,
    generate_tinier_model,
    generate_mobile_net,
    generate_fast_small_mobile_net,
    generate_mobile_net_v2,
    generate_small_mobile_net_v2,
    generate_mini_mobile_net_v2,
    generate_micro_mobile_net_v2,
    generate_little_mobile_net_v2,
    generate_little_mobile_net_v2_fast_upproj,
    generate_fast_puny_mobile_net_v2,
)
from data import get_item
from metrics import rmse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help='Options: s2d | smaller | even | tiny')
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
elif args.model == 'even_upproj':
    model = generate_even_smaller_upproj_model(args.normalized)
elif args.model == 'tiny':
    model = generate_tiny_model(args.normalized)
elif args.model == 'tiny_resnet_deconv':
    model = generate_tiny_resnet_deconv_model(args.normalized)
elif args.model == 'tiny_upproj':
    model = generate_tiny_upproj_model(args.normalized)
elif args.model == 'tiny_half_upproj':
    model = generate_tiny_half_upproj_model(args.normalized)
elif args.model == 'tiny_fast_upproj':
    model = generate_tiny_fast_upproj_model(args.normalized)
elif args.model == 'tiny_faster_upproj':
    model = generate_tiny_faster_upproj_model(args.normalized)
elif args.model == 'teeny_fast_upproj':
    model = generate_teeny_fast_upproj_model(args.normalized)
elif args.model == 'tinier':
    model = generate_tinier_model(args.normalized)
elif args.model == 'mn':
    model = generate_mobile_net(args.normalized)
elif args.model == 'fast_small_mn':
    model = generate_fast_small_mobile_net(args.normalized)
elif args.model == 'mn2':
    model = generate_mobile_net_v2(args.normalized)
elif args.model == 'small_mn2':
    model = generate_small_mobile_net_v2(args.normalized)
elif args.model == 'mini_mn2':
    model = generate_mini_mobile_net_v2(args.normalized)
elif args.model == 'micro_mn2':
    model = generate_micro_mobile_net_v2(args.normalized)
elif args.model == 'little_mn2':
    model = generate_little_mobile_net_v2(args.normalized)
elif args.model == 'little_mn2_fast_upproj':
    model = generate_little_mobile_net_v2_fast_upproj(args.normalized)
elif args.model == 'fast_puny_mn2':
    model = generate_fast_puny_mobile_net_v2(args.normalized)
model.load_weights(args.checkpoint_path)

input, target = get_item(args.input_path, args.normalized, False)
input = tf.expand_dims(input, 0)
target = tf.expand_dims(target, 0)

pred = model(input, training=False)
print(f'rmse: {rmse(target, pred)}')

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
cmap = 'plasma_r'
# Max depth is 1 (no units) if normalized, otherwise 4m by L5 sensor limits
if args.normalized:
    fig, ax = plt.subplots(2, 3)
    ax[0][0].imshow(rgb1)
    ax[0][1].imshow(dense1, cmap=cmap, vmin=0, vmax=vmax)
    ax[0][2].imshow(pred1, cmap=cmap, vmin=0, vmax=vmax)
    ax[1][0].imshow(rgb2)
    ax[1][1].imshow(dense2, cmap=cmap, vmin=0, vmax=vmax)
    ax[1][2].imshow(pred2, cmap=cmap, vmin=0, vmax=vmax)
else:
    fig, ax = plt.subplots(2, 4)
    ax[0][0].imshow(rgb1)
    ax[0][1].imshow(sparse1, cmap=cmap, vmin=0, vmax=vmax)
    im = ax[0][2].imshow(dense1, cmap=cmap, vmin=0, vmax=vmax)
    ax[0][3].imshow(pred1, cmap=cmap, vmin=0, vmax=vmax)
    ax[1][0].imshow(rgb2)
    ax[1][1].imshow(sparse2, cmap=cmap, vmin=0, vmax=vmax)
    ax[1][2].imshow(dense2, cmap=cmap, vmin=0, vmax=vmax)
    ax[1][3].imshow(pred2, cmap=cmap, vmin=0, vmax=vmax)
    fig.colorbar(im, ax=fig.get_axes())
plt.show()
