from torchvision import transforms, datasets
from torch.utils.data import DataLoader


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
