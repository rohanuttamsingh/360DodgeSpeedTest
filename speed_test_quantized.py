#!/usr/bin/python3
import argparse
import os
import numpy as np
import tensorflow as tf
from data import get_inputs_in_memory
from time import perf_counter

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', type=str)
parser.add_argument('-d', '--data_path', type=str)
parser.add_argument('-n', '--normalized', action='store_true', help='Set to normalize dataset')
args = parser.parse_args()

interpreter = tf.lite.Interpreter(model_path=args.model_path)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

inputs = get_inputs_in_memory(args.data_path, 'val', False, args.normalized, False)
inputs = [np.expand_dims(input, axis=0).astype(np.float32) for input in inputs]
ds_size = len(inputs)

start_time = perf_counter()
for input in inputs:
    interpreter.set_tensor(input_index, input)
    interpreter.invoke()
end_time = perf_counter()

total_seconds = end_time - start_time
average_seconds = total_seconds / ds_size
average_hertz = 1 / average_seconds

print(f'Total Seconds: {total_seconds}')
print(f'Average Seconds: {average_seconds}')
print(f'Average Hertz: {average_hertz}')
