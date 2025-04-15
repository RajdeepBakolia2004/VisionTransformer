import numpy as np
from Data import data
import torch
import pickle
def torch_to_numpy(dataset):
    """
    Converts PyTorch dataset into NumPy array.
    
    Args:
        dataset: PyTorch dataset
    
    Returns:
        NumPy array of shape (N, H, W, C)
    """
    with open("data.pkl","rb") as f:
        data, label = pickle.load(f)
    return data, label
'''
    N = len(dataset)
    data = torch.cat([dataset[i][0].unsqueeze(0) for i in range(N)], dim=0)
    label = torch.tensor([dataset[i][1] for i in range(N)])
    with open("data.pkl","wb") as f:
        pickle.dump((data.numpy(),label.numpy()), f)
        
         
    return data.numpy(), label.numpy()
'''
def extract_patches(images, patch_size=2):
    """
    Splits images into non-overlapping patches of given size.
    
    Args:
        images: NumPy array of shape (N, 32, 32, 3) [Batch of images]
        patch_size: Size of each square patch (default 2x2)
    
    Returns:
        Patches: NumPy array of shape (N, num_patches, patch_size, patch_size, 3)
    """
    N, H, W, C = images.shape  # Batch, Height, Width, Channels
    num_patches = (H // patch_size) * (W // patch_size)  # Total patches per image
    
    # Reshape into patches
    patches = images.reshape(N, H//patch_size, patch_size, W//patch_size, patch_size, C)
    
    # Rearrange axes to group patches together
    patches = patches.transpose(0, 1, 3, 2, 4, 5).reshape(N, num_patches, patch_size, patch_size, C)
    return patches

def patch_image(patch_size = 2):
    
    train_dataset, test_dataset, _, _ = data()
    x_train, y_train = torch_to_numpy(train_dataset)
    x_train = x_train.transpose(0, 2, 3, 1)  # (N, 32, 32, 3) -> (N, 32, 32, 3)
    patches = extract_patches(x_train, patch_size)
    return patches

if __name__ == "__main__":
    train_dataset, test_dataset, _, _ = data()
    x_train, y_train = torch_to_numpy(train_dataset)
    x_train = x_train.transpose(0, 2, 3, 1)  # (N, 32, 32, 3) -> (N, 32, 32, 3)
    patches = extract_patches(x_train, patch_size=2)
    print(f"Original Images Shape: {x_train.shape}")  # (N, 32, 32, 3)
    print(f"Patches Shape: {patches.shape}")  # (N, 256, 2, 2, 3)