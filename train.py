#!/usr/bin/python3

import argparse
import itertools
import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
from tqdm import tqdm
from models import Generator
from models import Discriminator
from utils import ImagePool
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from utils import save_image
from utils import GANLoss
from utils import set_requires_grad
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=2, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=400, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.input_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Lossess
criterion_GAN = GANLoss('lsgan').cuda()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=opt.lr,betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(itertools.chain(netD_A.parameters(), netD_B.parameters()), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D,lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
# target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
# target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_A_pool = ImagePool()
fake_B_pool = ImagePool()

# Dataset loader
transforms_ = [# transforms.Resize((int(opt.size * 1.12), int(opt.size * 1.12)), Image.BICUBIC),
               transforms.RandomCrop((opt.size, opt.size)),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu,  pin_memory=True)

dataloader = list(dataloader)
# Loss plot
# logger = Logger(opt.n_epochs, len(dataloader))
###################################

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in tqdm(enumerate(dataloader)):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        # forward
        fake_B = netG_A2B(real_A)  # G_A(A)
        rec_A = netG_B2A(fake_B)  # G_B(G_A(A))
        fake_A = netG_B2A(real_B)  # G_B(B)
        rec_B = netG_A2B(fake_A)  # G_A(G_B(B))

        ############# G_A and G_B  #################
        set_requires_grad([netD_A, netD_B], False)  # Ds require no gradients when optimizing Gs
        optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero

        # calculate gradients for G_A and G_B
        lambda_idt = 0.5
        lambda_A = 10
        lambda_B = 10
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            idt_A = netG_A2B(real_B)
            loss_idt_A = criterion_identity(idt_A, real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            idt_B = netG_B2A(real_A)
            loss_idt_B = criterion_identity(idt_B, real_A) * lambda_A * lambda_idt
        else:
            loss_idt_A = 0
            loss_idt_B = 0
            
        # GAN loss D_A(G_A(A))
        loss_G_A = criterion_GAN(netD_A(fake_B), True)
        # GAN loss D_B(G_B(B))
        loss_G_B = criterion_GAN(netD_B(fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        loss_cycle_A = criterion_cycle(rec_A, real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        loss_cycle_B = criterion_cycle(rec_B, real_B) * lambda_B
        # combined loss and calculate gradients
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        loss_G.backward()
        optimizer_G.step()  # update G_A and G_B's weights
        ###################################

        def backward_D_basic(netD, real, fake):
            """Calculate GAN loss for the discriminator

            Parameters:
                netD (network)      -- the discriminator D
                real (tensor array) -- real images
                fake (tensor array) -- images generated by a generator

            Return the discriminator loss.
            We also call loss_D.backward() to calculate the gradients.
            """
            # Real
            pred_real = netD(real)
            loss_D_real = criterion_GAN(pred_real, True)
            # Fake
            pred_fake = netD(fake.detach())
            loss_D_fake = criterion_GAN(pred_fake, False)
            # Combined loss and calculate gradients
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            return loss_D

        ############# D_A and D_B  #################
        set_requires_grad([netD_A, netD_B], True)
        optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        # calculate gradients for D_A
        fake_B_tmp = fake_B
        fake_B = fake_B_pool.query(fake_B)
        loss_D_A = backward_D_basic(netD_A, real_B, fake_B)
        fake_A_tmp = fake_A
        fake_A = fake_A_pool.query(fake_A)
        loss_D_B = backward_D_basic(netD_B, real_A, fake_A)
        optimizer_D.step()  # update D_A and D_B's weights

        images = {'real_A': real_A, 'fake_B': fake_B_tmp, "recovered_A": rec_A,
                  'real_B': real_B, 'fake_A': fake_A_tmp, "recovered_B": rec_B}

        # # Progress report (http://localhost:8097)
        # logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_idt_A + loss_idt_B),
        #             'loss_G_GAN': (loss_G_A + loss_G_B),
        #             'loss_G_cycle': (loss_cycle_A + loss_cycle_B), 'loss_D': (loss_D_A + loss_D_B)},
        #            images=images)

        if i % 50 == 0:
            row1 = torch.cat((images["real_A"].cpu(), images["fake_B"].cpu().data, images["recovered_A"].cpu().data), 3)
            row2 = torch.cat((images["real_B"].cpu(), images["fake_A"].cpu().data, images["recovered_B"].cpu().data), 3)
            result = torch.cat((row1, row2), 2)
            res_path = os.path.join("/home/lyc/code/HaoL/LDH/CycleGAN/log/train/syn_sunny2foggy/vis", '{:02d}_{:05d}.png'.format(epoch, i))

            save_image(result, res_path, normalize=True)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), '/home/lyc/code/HaoL/LDH/CycleGAN/log/train/syn_sunny2foggy/output/netG_A2B_{}.pth'.format(epoch))
    torch.save(netG_B2A.state_dict(), '/home/lyc/code/HaoL/LDH/CycleGAN/log/train/syn_sunny2foggy/output/netG_B2A_{}.pth'.format(epoch))
    torch.save(netD_A.state_dict(), '/home/lyc/code/HaoL/LDH/CycleGAN/log/train/syn_sunny2foggy/output/netD_A_{}.pth'.format(epoch))
    torch.save(netD_B.state_dict(), '/home/lyc/code/HaoL/LDH/CycleGAN/log/train/syn_sunny2foggy/output/netD_B_{}.pth'.format(epoch))
###################################
