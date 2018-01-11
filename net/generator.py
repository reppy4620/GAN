import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, nz=100, nfilter=64, nChannels=4):

        """
        :param nz: input dimension default=100
        :param nfilter: filter size default=64
        :param nChannel: output dimension, this network out color
                         as this channel is 3
        """

        super(Generator, self).__init__()

        # input : 100
        # output : nfilter*8 * 4 * 4
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=nz,
                out_channels=nfilter*16,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(nfilter*16),
            nn.ELU(inplace=True)
        )

        # input : nfilter*8 * 4 * 4
        # output : nfilter*4 * 8 * 8
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=nfilter*16,
                out_channels=nfilter*8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(nfilter*8),
            nn.ELU(inplace=True)
        )

        # input : nfilter*4 * 8 * 8
        # output : nfilter*4 * 16 * 16
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=nfilter*8,
                out_channels=nfilter*4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(nfilter*4),
            nn.ELU(inplace=True)
        )

        # input : nfilter*4 * 16 * 16
        # output : nfilter*4 * 32 * 32
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=nfilter*4,
                out_channels=nfilter*2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(nfilter*2),
            nn.ELU(inplace=True)
        )

        # input : nfilter * 32 * 32
        # output : nfilter * 64 * 64
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=nfilter*2,
                out_channels=nfilter,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(nfilter),
            nn.ELU(inplace=True)
        )

        # input : nfilter * 64 * 64
        # output : nfilter * 128 * 128
        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=nfilter,
                out_channels=nChannels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.Tanh()
        )

    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)

        return out
