import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])


def load_img(filepath):
    img = Image.open(filepath) #.convert('RGB')
    #R, G, B = img.split()
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, image_dir2, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames  = [join(image_dir , x) for x in listdir(image_dir ) if is_image_file(x)]
        self.image_filenames2 = [join(image_dir2, x) for x in listdir(image_dir2) if is_image_file(x)]
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input  = load_img(self.image_filenames2[index])
        target = load_img(self.image_filenames [index])
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)
