import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as uts
import sys
import os
import math

from torch import FloatTensor
from torch.autograd import Variable
from net.generator import Generator
from net.discriminator import Discriminator
from utils import get_loader, winnow


class Manager:

    def __init__(self, path, image_size, batch_size, nc):
        self.g = Generator(nChannels=nc).cuda()
        self.d = Discriminator(nChannels=nc, isLSGAN=False).cuda()
        self.opt_g = optim.Adam(params=self.g.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.opt_d = optim.Adam(params=self.d.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.data_loader = get_loader(path, image_size, batch_size, num_workers=4)
        self.batch_size = batch_size
        self.image_size = image_size
        self.nc = nc

        if not os.path.isdir('Data_and_Results/DCGAN_model'):
            os.mkdir('Data_and_Results/DCGAN_model')
        if not os.path.isdir('Data_and_Results/DCGAN_Result'):
            os.mkdir('Data_and_Results/DCGAN_Result')

    def train(self):
        noise = Variable(FloatTensor(self.batch_size, 100, 1, 1)).cuda()
        real = Variable(FloatTensor(self.batch_size, self.nc, self.image_size, self.image_size)).cuda()
        label = Variable(FloatTensor(self.batch_size)).cuda()
        nepoch = 1000
        real_label, fake_label = 1, 0
        bce = nn.BCELoss()

        for epoch in range(1, nepoch+1):
            for i, data in enumerate(self.data_loader):

                """Gradient of Discriminator"""
                self.d.zero_grad()
                images, _ = data

                real.data.resize_(images.size()).copy_(images)
                label.data.resize_(images.size(0)).fill_(real_label)

                # train Discriminator with real image
                output = self.d(real)
                errD_r = bce(output, label)
                errD_r.backward()

                # train Discriminator with fake image
                label.data.fill_(fake_label)
                noise.data.resize_(images.size(0), 100, 1, 1)
                noise.data.normal_(0, 1)

                # generate fake image
                fake = self.g(noise)

                # train
                output = self.d(fake.detach())
                errD_f = bce(output, label)
                errD_f.backward(retain_graph=True)

                errD = errD_r + errD_f
                self.opt_d.step()

                """Gradient of Generator"""
                self.g.zero_grad()
                label.data.fill_(real_label)
                output = self.d(fake)
                errG = bce(output, label)
                errG.backward()
                self.opt_g.step()

                """Output Log"""
                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%2d/%2d] Iter[%5d/%5d] Loss(D): %.4f Loss(G): %.4f'
                                 % (epoch, nepoch, i, len(self.data_loader), errD, errG))
                sys.stdout.flush()

                """Visualize"""
                if i % 10 == 0:
                    f_noise = Variable(FloatTensor(self.batch_size, 100, 1, 1).normal_(0, 1)).cuda()
                    f_fake = self.g(f_noise)
                    dir = 'Data_and_Results/DCGAN_Result/{0}_{1}.jpg'.format(epoch, i)
                    print(' | Saving result')
                    uts.save_image(
                        tensor=f_fake.data,
                        filename=dir,
                        nrow=int(math.sqrt(self.batch_size)),
                        normalize=True
                    )
            # save the model
            torch.save(self.g, 'Data_and_Results/DCGAN_model/net_g.pt')
            torch.save(self.d, 'Data_and_Results/DCGAN_model/net_d.pt')


if __name__ == '__main__':
    path = 'data'
    image_size = 128
    batch_size = 100
    nc = 3
    winnow('data/face', image_size)
    lsgan = Manager(path, image_size, batch_size, nc)
    lsgan.train()
