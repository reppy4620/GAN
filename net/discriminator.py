import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, nfilter=64, nChannels=3):

        """
        :param nfilter: filter size of Convolution layer
        :param nChannels: input dimension
        """

        super(Discriminator, self).__init__()

        # input : nChannels * 128 * 128
        # output : nfilter * 64 * 64
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=nChannels,
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
        # output : nfilter * 32 * 32
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=nfilter,
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
        # output : nfilter * 16 * 16
        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=nfilter*2,
                out_channels=nfilter*4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(nfilter*4),
            nn.ELU(inplace=True)
        )

        # input : nfilter * 8 * 8
        # output : nfilter * 4 * 4
        self.layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=nfilter*4,
                out_channels=nfilter*8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(nfilter*8),
            nn.ELU(inplace=True)
        )

        # input : nfilter * 4 * 4
        # output : result of judge Real or Fake
        self.layer5 = nn.Sequential(
            nn.Conv2d(
                in_channels=nfilter*8,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            )
        )

    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        return out.view(-1, 1)
