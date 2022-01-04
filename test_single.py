#!/usr/bin/python3

import argparse
import sys
import os
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models import Generator
from datasets import ImageDataset
from datasets import SingleDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='/home/lyc/data/ll/dataset/gta2cityscapes/',
                    help='root directory of the dataset')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--model_dir', type=str, help='模型存放的文件夹')
parser.add_argument('--which_epoch', type=int, help='测试哪个epoch')
parser.add_argument('--res_dir', type=str, default=r'res')
parser.add_argument('--direction', type=str, default=r'A2B')

opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
G = Generator(opt.input_nc, opt.output_nc)
if opt.cuda:
    G = G.cuda()
if opt.direction == 'A2B':
    model_path = os.path.join(opt.model_dir, 'netG_A2B_{}.pth'.format(opt.which_epoch))
else:
    model_path = os.path.join(opt.model_dir, 'netG_B2A_{}.pth'.format(opt.which_epoch))

G.load_state_dict(torch.load(model_path))
G.eval()

# Dataset loader
transforms_ = [
    # transforms.Resize([1024, 2048], Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

dataloader = DataLoader(SingleDataset(opt.dataroot, transforms_=transforms_),
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)

###### Testing######

# Create output dirs if they don't exist
real_dir = os.path.join(opt.res_dir, "ori")
fake_dir = os.path.join(opt.res_dir, "trans")
if not os.path.exists(real_dir):
    os.makedirs(real_dir)
if not os.path.exists(fake_dir):
    os.makedirs(fake_dir)

for i, images in enumerate(dataloader):
    # Set model input
    real = images
    if opt.cuda:
        real = real.cuda()
    with torch.no_grad():
        fake = G(real)

    fake_path = os.path.join(fake_dir, '{:05d}.png'.format(i))
    # real_path = os.path.join(real_dir, '{:05d}.png'.format(i))

    save_image(fake, fake_path, normalize=True)
    # save_image(real, real_path, normalize=True)

    sys.stdout.write('\rGenerated images %04d of %04d' % (i + 1, len(dataloader)))

sys.stdout.write('\n')
###################################
