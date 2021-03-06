import torch
import torch.optim as optim
import torchvision.utils as uts
import sys
import os
import math

from torch import FloatTensor
from torch.autograd import Variable
from net.generator import Generator
from net.discriminator import Discriminator
from utils import get_loader, winnow, load_model


class Manager:

    def __init__(self, path, image_size, batch_size, nc):
        self.g = Generator(nChannels=nc).cuda()
        self.d = Discriminator(nChannels=nc).cuda()
        self.opt_g = optim.Adam(params=self.g.parameters(), lr=2e-4, betas=(0.1, 0.999))
        self.opt_d = optim.Adam(params=self.d.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.data_loader = get_loader(path, image_size, batch_size, num_workers=4)
        self.batch_size = batch_size
        self.image_size = image_size
        self.nc = nc

        if not os.path.isdir('Data_and_Results/LSGAN_model'):
            os.mkdir('Data_and_Results/LSGAN_model')
        if not os.path.isdir('Data_and_Results/LSGAN_Result'):
            os.mkdir('Data_and_Results/LSGAN_Result')

    def train(self):
        noise = Variable(FloatTensor(self.batch_size, 100, 1, 1)).cuda()
        real = Variable(FloatTensor(self.batch_size, self.nc, self.image_size, self.image_size)).cuda()
        label = Variable(FloatTensor(self.batch_size)).cuda()
        nepoch = 100000
        real_label, fake_label = 1, 0

        def loss_func(output, label):
            return 0.5 * torch.mean((output-label)**2)

        for epoch in range(1, nepoch+1):
            for i, data in enumerate(self.data_loader):

                """Gradient of Discriminator"""
                self.d.zero_grad()
                images, _ = data

                real.data.resize_(images.size()).copy_(images)
                label.data.resize_(images.size(0)).fill_(real_label)

                # train Discriminator with real image
                output = self.d(real)
                errD_r = loss_func(output, label)
                errD_r.backward()

                # train Discriminator with fake image
                label.data.fill_(fake_label)
                noise.data.resize_(images.size(0), 100, 1, 1)
                noise.data.normal_(0, 1)

                # generate fake image
                fake = self.g(noise)

                # train
                output = self.d(fake.detach())
                errD_f = loss_func(output, label)
                errD_f.backward(retain_graph=True)

                errD = errD_r + errD_f
                self.opt_d.step()

                """Gradient of Generator"""
                self.g.zero_grad()
                label.data.fill_(real_label)
                output = self.d(fake)
                errG = loss_func(output, label)
                errG.backward()
                self.opt_g.step()

                """Output Log"""
                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%2d/%2d] Iter[%5d/%5d] Loss(D): %.4f Loss(G): %.4f'
                                 % (epoch, nepoch, i, len(self.data_loader), errD, errG))
                sys.stdout.flush()

                """Visualize"""
                if i % 100 == 0:
                    f_noise = Variable(FloatTensor(self.batch_size, 100, 1, 1).normal_(0, 1)).cuda()
                    f_fake = self.g(f_noise)
                    dir = 'Data_and_Results/LSGAN_Result/{0}_{1}.jpg'.format(epoch, i)
                    print(' | Saving result')
                    uts.save_image(
                        tensor=f_fake.data,
                        filename=dir,
                        nrow=int(math.sqrt(self.batch_size)),
                        normalize=True
                    )
            # save the model
            torch.save(self.g, 'Data_and_Results/LSGAN_model/net_g.pt')
            torch.save(self.d, 'Data_and_Results/LSGAN_model/net_d.pt')

            if epoch % 50 == 0:
                torch.save(self.g, 'Data_and_Results/LSGAN_model/net_g_{}.pt'.format(epoch))
                torch.save(self.d, 'Data_and_Results/LSGAN_model/net_d_{}.pt'.format(epoch))

    def train_with_model(self):
        self.g = load_model('LSGAN', 'g')
        self.d = load_model('LSGAN', 'd')
        self.train()


if __name__ == '__main__':
    path = 'Data_and_Results/data/face'
    image_size = 128
    batch_size = 128
    nc = 3
    # winnow(path, image_size)
    lsgan = Manager('Data_and_Results/winnowed', image_size, batch_size, nc)
    lsgan.train_with_model()
