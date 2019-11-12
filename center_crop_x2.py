from __future__ import print_function
import argparse
import random
import torch
from torch.autograd import Variable
from PIL import Image
import tensorflow as tf

import numpy as np
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, RandomCrop

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input_image', type=str, required=True, help='input image to use')
parser.add_argument('--output_filename', type=str, help='where to save the output image')
parser.add_argument('--x', type=int, required=True, help='init x position')
parser.add_argument('--y', type=int, required=True, help='init y position')
opt = parser.parse_args()

print(opt)
img = Image.open(opt.input_image).convert('RGB')

ow, oh = 256, 256
w , h  = img.size
th, tw = 128, 128

tx = opt.x
ty = opt.y

crop_img = img.crop((tx, ty, tx+tw, ty+th))

resized_img = crop_img.resize((256, 256), Image.BICUBIC)

dir = opt.output_filename.split('.')[0]
base = opt.output_filename.split('.')[1]
ext  = opt.output_filename.split('.')[2]
out_name = "." + dir + base + "_x2resized" +"." + ext

print('output image saved to ', out_name)

resized_img.save(out_name)
