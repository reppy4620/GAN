import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as uts
import sys
import os

from torch import FloatTensor
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
from net.generator import Generator
from net.discriminator import Discriminator


class Manager:

    def __init__(self, path, image_size, batch_size, nc):
        self.g = Generator(nChannels=nc).cuda()
        self.d = Discriminator(nChannels=nc).cuda()
        self.opt_g = optim.Adam(params=self.g.parameters(), lr=2e-4)
        self.opt_d = optim.Adam(params=self.d.parameters(), lr=1e-5)
        self.batch_size = batch_size
        self.image_size = image_size
        self.nc = nc

        data_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.ToTensor()
        ])
        dataset = datasets.ImageFolder(
            root=path,
            transform=data_transform
        )
        self.data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )

        if not os.path.isdir('models'):
            os.mkdir('models')
        if not os.path.isdir('Result'):
            os.mkdir('Result')

    def train(self):
        noise = Variable(FloatTensor(self.batch_size, 100, 1, 1)).cuda()
        real = Variable(FloatTensor(self.batch_size, self.nc, self.image_size, self.image_size)).cuda()
        label = Variable(FloatTensor(self.batch_size)).cuda()
        nepoch = 1000
        real_label, fake_label = 1, 0
        bce = nn.BCELoss().cuda()

        def loss_func(output, label):
            return 0.5 * torch.mean((output-label)**2)

        for epoch in range(1, nepoch+1):
            for i, (images, _) in enumerate(self.data_loader):

                """Gradient of Discriminator"""
                self.d.zero_grad()

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
                                 % (epoch, nepoch, i, len(self.data_loader), errD.data[0], errG.data[0]))
                sys.stdout.flush()

                """Visualize"""
                if i % 10 == 0:
                    dir = 'Result/{0}_{1}.jpg'.format(epoch, i)
                    print(' | Saving result')
                    uts.save_image(
                        fake.data,
                        dir,
                        normalize=True
                    )
            torch.save(self.g.state_dict(), 'models/net_g.pth')
            torch.save(self.d.state_dict(), 'models/net_d.pth')


if __name__ == '__main__':
    path = 'data'
    image_size = 128
    batch_size = 64
    nc = 3
    lsgan = Manager(path, image_size, batch_size, nc)
    lsgan.train()
