from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
import tensorflow as tf

import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input_image', type=str, required=True, help='input image to use')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--output_filename', type=str, help='where to save the output image')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--network', type=float, default='rtsr', help='Network Setlection')

opt = parser.parse_args()

print(opt)
img = Image.open(opt.input_image)#.convert('RGB')
r, g, b = img.split()

#old_model
#y, cb, cr = img.split()
#model = torch.load(opt.model)
#img_to_tensor = ToTensor()
#input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])

model = torch.load(opt.model)
img_to_tensor = ToTensor()
input_r = img_to_tensor(r).view(1, -1, r.size[1], r.size[0])
input_g = img_to_tensor(g).view(1, -1, r.size[1], r.size[0])
input_b = img_to_tensor(b).view(1, -1, r.size[1], r.size[0])
img     = torch.cat((input_r, input_g, input_b), 1)

if opt.cuda:
    model = model.cuda()
    #input = input.cuda()
    input = img.cuda()

out = model(input)
out = out.cpu()

#print(out.size())


#out_img_y = out[0].detach().numpy()
#out_img_y *= 256.0
#out_img_y = out_img_y.clip(0, 255)
#out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
#out_img2_y = out_img_y.resize(y.size, Image.LANCZOS)

out_r = out[0, 0].detach().numpy()
out_g = out[0, 1].detach().numpy()
out_b = out[0, 2].detach().numpy()
#print(out_r.size())

out_img_r = out_r * 255.0 + 0.5
out_img_r = out_img_r.clip(0, 255)
out_img_r = Image.fromarray(np.uint8(out_img_r), mode='L')

out_img_g = out_g * 255.0 + 0.5
out_img_g = out_img_g.clip(0, 255)
out_img_g = Image.fromarray(np.uint8(out_img_g), mode='L')

out_img_b = out_b * 255.0 + 0.5
out_img_b = out_img_b.clip(0, 255)
out_img_b = Image.fromarray(np.uint8(out_img_b), mode='L')

out_img2_r = out_img_r.resize(r.size, Image.LANCZOS)
out_img2_g = out_img_g.resize(g.size, Image.LANCZOS)
out_img2_b = out_img_b.resize(b.size, Image.LANCZOS)

out_img = Image.merge('RGB', [out_img2_r, out_img2_g, out_img2_b])

out_img.save(opt.output_filename)
print('output image saved to ', opt.output_filename)
