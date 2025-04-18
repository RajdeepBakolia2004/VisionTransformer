import os
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from pathlib import Path

def save_mnist_images(data_dir, save_dir, train=True):
    # Load MNIST from raw files
    dataset = MNIST(
        root=data_dir,
        train=train,
        download=True,
        transform=transforms.ToTensor()
    )
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    split = 'train' if train else 'test'
    print(f"Saving {split} images...")

    for i, (img, label) in enumerate(loader):
        label_dir = Path(save_dir) / split / str(label.item())
        label_dir.mkdir(parents=True, exist_ok=True)
        save_path = label_dir / f"{i}.png"
        save_image(img, save_path)

    print(f"Done saving {split} images to {save_dir}/{split}")

# Fixed path (point to folder that contains 'raw' and/or 'processed')
data_dir = '/home/rajdeep/UMC203project/SimpleTransformer/MNIST'
save_dir = '/home/rajdeep/UMC203project/SimpleTransformer/extract'

save_mnist_images(data_dir, save_dir, train=True)
save_mnist_images(data_dir, save_dir, train=False)


