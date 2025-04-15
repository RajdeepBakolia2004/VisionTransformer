from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,TensorDataset
import random
import matplotlib.pyplot as plt
import torch

DATASET_PATH = "cifar10_images"
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def train_dataset(patch_size):
    
    train_dataset = ImageFolder(root="cifar10_images/train", transform=transform)
    train_patches, train_labels = transform_dataset_to_patches(train_dataset, patch_size)
    train_data = TensorDataset(train_patches, train_labels)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    return train_loader

def save_random_images(count=5):
    dataset = ImageFolder(root="cifar10_images/train", transform=transform)
    for i in range(count):
        idx = random.randint(0, len(dataset) - 1)
        img, label = dataset[idx]
        img = img.permute(1, 2, 0) * 0.5 + 0.5  
        plt.imshow(img)
        plt.title(dataset.classes[label])
        plt.axis('off')
        plt.savefig(f"sample_{i}.png")

def test_dataset(patch_size):
    test_dataset = ImageFolder(root="cifar10_images/test", transform=transform)
    print(test_dataset.classes)
    test_patches, test_labels = transform_dataset_to_patches(test_dataset, patch_size)
    test_data = TensorDataset(test_patches, test_labels)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    return test_loader


def extract_patches(img_tensor, patch_size):
    C, H, W = img_tensor.shape
    P = patch_size
    
    patches = img_tensor.unfold(1, P, P).unfold(2, P, P)  # [C, H//P, W//P, P, P]
    patches = patches.permute(1, 2, 0, 3, 4)               # [H//P, W//P, C, P, P]
    patches = patches.reshape(-1, C * P * P)              # [N, P*P*C]
    return patches

def transform_dataset_to_patches(dataset, patch_size):
    all_patches = []
    all_labels = []

    for img, label in dataset:
        patches = extract_patches(img, patch_size)  # shape: [N, P*P*C]
        all_patches.append(patches)
        all_labels.append(label)

    # Convert list of [N, D] tensors to one big tensor: [B, N, D]
    all_patches = torch.stack(all_patches)  # shape: [B, N, patch_dim]
    all_labels = torch.tensor(all_labels)

    return all_patches, all_labels

if __name__ == "__main__":
    # Example usage
    patch_size = 4
    train_loader = train_dataset(patch_size)
    test_loader = test_dataset(patch_size)
    save_random_images()
    print("Train and test datasets created successfully.")
    print("Random images saved successfully.")
    

