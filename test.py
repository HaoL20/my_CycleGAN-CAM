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

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='/home/lyc/data/ll/dataset/gta2cityscapes/',
                    help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
# parser.add_argument('--generator_A2B', type=str, default=r'output/netG_A2B_65.pth', help='A2B generator checkpoint file')
# parser.add_argument('--generator_B2A', type=str, default=r'output/netG_B2A_65.pth', help='B2A generator checkpoint file')
parser.add_argument('--model_dir', type=str, help='模型存放的文件夹')
parser.add_argument('--which_epoch', type=int, help='测试哪个epoch')
parser.add_argument('--res_dir', type=str, default=r'res')

opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()

# Load state dicts
model_A2B_path = os.path.join(opt.model_dir,'netG_A2B_{}.pth'.format(opt.which_epoch))
model_B2A_path = os.path.join(opt.model_dir,'netG_B2A_{}.pth'.format(opt.which_epoch))
netG_A2B.load_state_dict(torch.load(model_A2B_path))
netG_B2A.load_state_dict(torch.load(model_B2A_path))

# Set model's test.py mode
netG_A2B.eval()
# netG_B2A.eval()

'''# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
'''
# Dataset loader
transforms_ = [
    transforms.Resize([1024, 2048], Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'),
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
# dataloader = DataLoader(CityscapeDataset(opt.dataroot, transforms_=transforms_),
#                         batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
###################################

###### Testing######

# Create output dirs if they don't exist
# res_A = '{}/A'.format(opt.res_dr)
# res_B = '{}/B'.format(opt.res_dr)
# res_A_real = '{}/A_real'.format(opt.res_dr)
# res_B_real = '{}/B_real'.format(opt.res_dr)
# if not os.path.exists(res_A):
#     os.makedirs(res_A)
# if not os.path.exists(res_B):
#     os.makedirs(res_B)
# if not os.path.exists(res_A_real):
#     os.makedirs(res_A_real)
# if not os.path.exists(res_B_real):
#     os.makedirs(res_B_real)

if not os.path.exists(opt.res_dir):
    os.makedirs(opt.res_dir)
print(opt.res_dir)
for i, batch in enumerate(dataloader):
    # Set model input
    # real_A = Variable(input_A.copy_(batch['A']))
    # real_B = Variable(input_B.copy_(batch['B']))
    real_A = batch['A']
    real_B = batch['B']
    if opt.cuda:
        real_A = batch['A'].cuda()
        # print(real_A.shape)
        real_B = batch['B'].cuda()
    with torch.no_grad():
        fake_B = netG_A2B(real_A)
        # print(fake_B.shape)
        fake_A = netG_B2A(real_B)
        recovered_A = netG_B2A(fake_B)
        # print(recovered_A.shape)
        recovered_B = netG_A2B(fake_A)


    # # Generate output
    # fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
    # fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)
    # real_A = 0.5 * (real_A + 1.0)
    # real_B = 0.5 * (real_B + 1.0)

    row1 = torch.cat((real_A.cpu(), fake_B.cpu().data, recovered_A.cpu().data), 3)
    row2 = torch.cat((real_B.cpu(), fake_A.cpu().data, recovered_B.cpu().data), 3)
    result = torch.cat((row1, row2), 2)
    res_path = os.path.join(opt.res_dir, '{:05d}.png'.format(i))
    save_image(result, res_path, normalize=True)


    # # Save image files
    # save_image(fake_A, os.path.join(res_A,'%04d.png' % (i + 1)))
    # save_image(fake_B, os.path.join(res_B,'%04d.png' % (i + 1)))
    # save_image(real_A, os.path.join(res_A_real,'%04d.png' % (i + 1)))
    # save_image(real_B, os.path.join(res_B_real,'%04d.png' % (i + 1)))

    sys.stdout.write('\rGenerated images %04d of %04d' % (i + 1, len(dataloader)))

sys.stdout.write('\n')
###################################
