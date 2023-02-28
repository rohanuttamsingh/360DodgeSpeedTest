#!/usr/bin/python3
import argparse
import os
import tensorflow as tf
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
from data import get_inputs_in_memory
from time import perf_counter

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help='Options: s2d | smaller | even | tiny')
parser.add_argument('-c', '--checkpoint_path', type=str, help='Path to model checkpoint')
parser.add_argument('-d', '--data_path', type=str, help='Path to diretory with input files')
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

inputs = get_inputs_in_memory(args.data_path, 'val', False, args.normalized, False)
inputs = [tf.expand_dims(input, axis=0) for input in inputs]
ds_size = len(inputs)

start_time = perf_counter()
for input in inputs:
    pred = model(input, training=False)
end_time = perf_counter()

total_seconds = end_time - start_time
average_seconds = total_seconds / ds_size
average_hertz = 1 / average_seconds

print(f'Total Seconds: {total_seconds}')
print(f'Average Seconds: {average_seconds}')
print(f'Average Hertz: {average_hertz}')
