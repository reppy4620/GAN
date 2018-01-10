import torch
import torch.optim as optim

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
        noise = Variable(torch.FloatTensor(batch_size, 100, 1, 1)).cuda()
        real = Variable(torch.FloatTensor(batch_size, nc, image_size, image_size))
        label = Variable(torch.FloatTensor(batch_size)).cuda()
        real_label, fake_label = 1, 0

        for epoch in range(1, 101):
            for i, (images) in enumerate(self.data_loader):
                self.d.zero_grad()

                real.data.resize_(images.size()).copy_(images)
                label.data.resize_(images.size(0)).fill_(real_label)

                output = self.d(real)
                errD_r = 0.5 * torch.mean((output - label)**2)
                errD_r.backward()

                label.data.fill_(fake_label)
                noise.data.resize_(images.size(0), 100, 1, 1)
                noise.data.normal_(0, 1)



