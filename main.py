import torch
import torch.optim as optim
import torchvision.utils as uts
import sys

from torch import FloatTensor
from torchvision import transforms
from torch.autograd import Variable
from data_loader import get_loader
from net.generator import Generator
from net.discriminator import Discriminator


class Manager:

    def __init__(self, path, image_size, batch_size):
        self.g = Generator().cuda()
        self.d = Discriminator().cuda()
        self.opt_g = optim.Adam(params=self.g.parameters(), lr=2e-4)
        self.opt_d = optim.Adam(params=self.d.parameters(), lr=1e-5)

        data_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.ToTensor()
        ])
        self.data_loader = get_loader(path, batch_size, data_transform)

    def train(self, batch_size, image_size, nc):
        noise = Variable(FloatTensor(batch_size, 100, 1, 1)).cuda()
        real = Variable(FloatTensor(batch_size, nc, image_size, image_size))
        label = Variable(FloatTensor(batch_size)).cuda()
        nepoch = 100
        real_label, fake_label = 1, 0

        def loss_func(outpu, label):
            return 0.5 * torch.mean((output-label)**2)

        for epoch in range(1, nepoch+1):
            for i, (images) in enumerate(self.data_loader):

                """Gradient of Discriminator"""
                self.d.zero_grad()

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
                output = self.d(fake)
                errD_f = loss_func(output, label)
                errD_f.backward()

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
                                 .format(epoch, nepoch, i, len(self.data_loader), errD.data[0], errG.data[0]))
                sys.stdout.flush()

                """Visualize"""
                if i % 10 == 0:
                    dir = 'Result/{0}_{1}.png'.format(epoch, i)
                    print('Saving result')
                    uts.save_image(
                        fake.data,
                        dir,
                        normalize=True
                    )
            torch.save(self.g.state_dict(), 'models/net_g.pth')
            torch.save(self.d.state_dict(), 'models/net_d.pth')


if __name__ == '__main__':
    path = 'ManyData'
    lsgan = Manager(path, 128, 100)
