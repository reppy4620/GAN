import os
from torch.utils import data
from PIL import Image


class ImageFolder(data.Dataset):

    def __init__(self, path, transform=None):
        self.image_path = list(map(lambda x: os.path.join(path, x), os.listdir(path)))
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_path[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image


def get_loader(image_path, batch_size, transform, num_workers=4):

    dataset = ImageFolder(image_path, transform)
    data_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return data_loader
