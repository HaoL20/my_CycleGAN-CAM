import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, transforms_label_ = None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        # self.transform_label = transforms.Compose(transforms_label_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        # self.files_A_label = sorted(glob.glob(os.path.join(root, '%s/A_label' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

        # self.files_B_50 = sorted(glob.glob(os.path.join(root, '%s/B_50' % mode) + '/*.*'))
        # self.files_B_50_label = sorted(glob.glob(os.path.join(root, '%s/B_50_label' % mode) + '/*.*'))
        self.adjust_num()

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index]))
        item_B = self.transform(Image.open(self.files_B[index]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def adjust_num(self):
        x_len = len(self.files_A)
        y_len = len(self.files_B)
        if x_len >= y_len:
            y_append_num = x_len - y_len
            y_append_list = [self.files_B[np.random.randint(y_len)]  for i in range(y_append_num)]
            self.files_B.extend(y_append_list)
        else:
            x_append_num = y_len - x_len
            x_append_list = [self.files_A[np.random.randint(x_len)]  for i in range(x_append_num)]
            self.files_A.extend(x_append_list)

class SingleDataset(Dataset):
    def __init__(self,root, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        # self.images = glob.glob(os.path.join(root,'leftImg8bit','val') + '/*/*.*')
        self.images = get_files(root)
        self.images_size = len(self.images)
        print(self.images_size)
    def __getitem__(self, index):
        image = self.transform(Image.open(self.images[index]))
        return image
    def __len__(self):
        return self.images_size


def get_files(dir):
    fileslist = []
    for root, dirs, files in os.walk(dir):
        for filename in files:
            fileslist.append(os.path.join(root, filename))
    return sorted(fileslist)

