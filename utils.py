from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import load
import os
import math
from PIL import Image
import shutil


def get_loader(path, image_size, batch_size, num_workers=4):

    data_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.ImageFolder(
        root=path,
        transform=data_transform
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return data_loader


def load_model(mode='LSGAN', g_or_d='g'):
    path = ''
    if mode == 'LSGAN':
        if g_or_d == 'g':
            path = 'Data_and_Results/LSGAN_model/net_g.pt'
        elif g_or_d == 'd':
            path = 'Data_and_Results/LSGAN_model/net_d.pt'
    elif mode == 'DCGAN':
        if g_or_d == 'g':
            path = 'Data_and_Results/DCGAN_model/net_g.pt'
        elif g_or_d == 'd':
            path = 'Data_and_Results/DCGAN_model/net_d.pt'
    else:
        print('Please choose mode "LSGAN" or "DCGAN"')
    model = load(path)
    return model


def rename(path):
    path_list = os.listdir(path)
    data_size = int(math.log10(len(path_list)) + 1)
    print(data_size)
    for i, path_ in enumerate(path_list):
        num = '{}'.format(i)
        os.rename(path + '/' + path_, path + '/' + num.zfill(data_size) + '.jpg')


def winnow(path, max_img_size):
    if not os.path.isdir('winnowed/imgs_{}'.format(max_img_size)):
        os.mkdir('winnowed/imgs_{}'.format(max_img_size))
    for i, path_ in enumerate(os.listdir(path), start=1):
        img = Image.open(path+'/'+path_)
        width, height = img.size
        if width > max_img_size or height > max_img_size:
            shutil.copy(path+'/'+path_, 'winnowed/imgs_{}'.format(max_img_size))
    print('Successfully winnowed')
    rename('winnowed/imgs_{}'.format(max_img_size))
    print('Successfully renamed')
