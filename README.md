# 360Dodge SpeedTest

## Requirements

* Python 3.8.10
* TensorFlow 2.6.0
* NumPy 1.21.4
* H5py 3.1.0

## Instructions

1. Download the `checkpoints.tar.gz` file [here](https://drive.google.com/file/d/1A24q0726HSZw8tz1nbYeVA9arAAtJ-ku/view?usp=sharing) 
1. Download the `data.tar.gz` file [here](https://drive.google.com/file/d/12TIkYaKux6Pfn0oQwr4bHxe9rLifOWF_/view?usp=sharing)
1. Unzip both files into this directory with `tar -xzvf checkpoints.tar.gz` and `tar -xzvf data.tar.gz`
1. Run `./speed_test.py -m smaller -c checkpoints/smaller/latest.ckpt -d data` and record the results
1. Run `./speed_test.py -m even -c checkpoints/even_smaller/latest.ckpt -d data` and record the results

