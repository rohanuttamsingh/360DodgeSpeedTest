#!/usr/bin/python3
import argparse
import os
import tensorflow as tf
from models import generate_s2d_model, generate_smaller_model, generate_even_smaller_model
from data import get_inputs_in_memory
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help='Options: s2d | smaller | even')
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
model.load_weights(args.checkpoint_path)

inputs = get_inputs_in_memory(args.data_path, 'val', False, args.normalized, False)
inputs = [tf.expand_dims(input, axis=0) for input in inputs]
ds_size = len(inputs)

start_time = datetime.now()
for input in inputs:
    pred = model(input, training=False)
end_time = datetime.now()

total_seconds = (end_time - start_time).total_seconds()
average_seconds = total_seconds / ds_size
average_hertz = 1 / average_seconds

print(f'Total Seconds: {total_seconds}')
print(f'Average Seconds: {average_seconds}')
print(f'Average Hertz: {average_hertz}')
