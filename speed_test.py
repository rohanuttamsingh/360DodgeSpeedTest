#!/usr/bin/python3
import argparse
import os
import tensorflow as tf
from models import generate_model
from data import dataset_generator, get_dataset_size
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('-checkpoint_path', type=str, help='Path to model checkpoint')
parser.add_argument('-data_path', type=str, help='Path to diretory with input files')
args = parser.parse_args()

model = generate_model()
model.load_weights(args.checkpoint_path)

ds = tf.data.Dataset.from_generator(
    dataset_generator,
    args=[args.data_path, '', False],
    output_types=(tf.float32, tf.float32),
    output_shapes=((228, 304, 8), (228, 304, 2))
)
ds_size = get_dataset_size(args.data_path, '')

start_time = datetime.now()
for input, target in ds.batch(1):
    pred = model(input, training=False)
end_time = datetime.now()

total_seconds = (end_time - start_time).total_seconds()
average_seconds = total_seconds / ds_size
average_hertz = 1 / average_seconds

print(f'Total Seconds: {total_seconds}')
print(f'Average Seconds: {average_seconds}')
print(f'Average Hertz: {average_hertz}')
