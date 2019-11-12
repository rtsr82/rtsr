from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor, RandomCrop

from dataset import DatasetFromFolder



def download_div2k(dest="dataset"):
    output_image_dir = join(dest, "DIV2K_train_HR/images")

    if not exists(output_image_dir):
        makedirs(dest)
        url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


def download_bsd300(dest="dataset"):
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        makedirs(dest)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])

def get_training_set(upscale_factor, sr_run):
    root_dir = download_div2k() #download_bsd300()
    train_dir  = join(root_dir, "train_crop_sr/", str(sr_run) )
    print(train_dir)
    train_dir2 = join(root_dir, "compressed_train_crop_png")
    print(train_dir2)
    crop_size = calculate_valid_crop_size(128, upscale_factor)

    return DatasetFromFolder(train_dir, train_dir2,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))


def get_test_set(upscale_factor, sr_run):
    root_dir = download_div2k() #download_bsd300() #
    test_dir = join(root_dir, "valid_sr/", str(sr_run) )
    test_dir2= join(root_dir, "compressed_valid_png")
    crop_size = calculate_valid_crop_size(128, upscale_factor)

    return DatasetFromFolder(test_dir, test_dir2,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))


