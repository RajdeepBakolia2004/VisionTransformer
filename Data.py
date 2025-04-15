import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from time import time


def data():
    # Define transforms (keeping original pixel values in [0,1])
    transform = transforms.Compose([
        transforms.ToTensor()  # Converts images to tensors without normalization
    ])

    # Load CIFAR-10 dataset from `ImageFolder`
    train_dataset = datasets.ImageFolder(root="cifar10_images/train", transform=transform)
    test_dataset = datasets.ImageFolder(root="cifar10_images/test", transform=transform)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    return train_dataset, test_dataset, train_loader, test_loader
    

if __name__ == "__main__":
    train_dataset, test_dataset, _, _ = data()
    print(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")
    print(f"Class names: {train_dataset.classes}")

'''
Epoch [1/10], Loss: 1.7175, Accuracy: 38.04%
Epoch [2/10], Loss: 1.3761, Accuracy: 50.74%
Epoch [3/10], Loss: 1.2350, Accuracy: 56.04%
Epoch [4/10], Loss: 1.1295, Accuracy: 59.61%
Epoch [5/10], Loss: 1.0381, Accuracy: 62.95%
Epoch [6/10], Loss: 0.9537, Accuracy: 66.15%
Epoch [7/10], Loss: 0.8720, Accuracy: 69.05%
Epoch [8/10], Loss: 0.8025, Accuracy: 71.68%
Epoch [9/10], Loss: 0.7429, Accuracy: 73.76%
Epoch [10/10], Loss: 0.7085, Accuracy: 75.09%
Model saved successfully!
'''