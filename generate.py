from torch.autograd import Variable
from torch import FloatTensor
import torchvision.utils as utils
import matplotlib.pyplot as plt
from torchvision.transforms import transforms

from utils import load_model


def eval():
    model = load_model(mode='LSGAN')
    noise = Variable(FloatTensor(1, 100, 1, 1).normal_(0, 1)).cuda()
    # for i in range(100):
    #     noise[0, i] = 1e-4
    plt.ion()
    output = model(noise)
    img = transforms.ToPILImage(output.data)
    print(type(img))
    path = 'test.jpg'
    print('Saving Image')
    utils.save_image(
        tensor=output.data,
        filename=path,
        normalize=True
    )


if __name__ == '__main__':
    eval()
