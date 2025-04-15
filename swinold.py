def extract_flattened_patches(img_tensor, patch_size=4):
    """
    img_tensor: Tensor of shape [3, 96, 96]
    Returns: Tensor of shape [24, 24, 48]
    """
    C, H, W = img_tensor.shape
    
    # Step 1: Unfold (non-overlapping 4x4 patches)
    patches = img_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    # Shape: [C, H//P, W//P, P, P] → [3, 24, 24, 4, 4]

    # Step 2: Permute to [24, 24, 3, 4, 4]
    patches = patches.permute(1, 2, 0, 3, 4)

    # Step 3: Flatten last three dims → [24, 24, 48]
    patches = patches.reshape(patches.shape[0], patches.shape[1], -1)

    return patches  # [24, 24, 48]

def extract_patches_from_dataset(dataset, patch_size=4):
    """
    dataset: ImageFolder dataset
    Returns: Tensor of shape [N, 24, 24, 48]
    """
    all_patches = []
    all_labels = [] 
    
    for img, label in dataset:
        patches = extract_flattened_patches(img, patch_size)
        all_patches.append(patches)
        all_labels.append(label)
    
    all_patches = torch.stack(all_patches)  # Shape: [N, 24, 24, 48]
    all_labels = torch.tensor(all_labels)
    
    return all_patches, all_labels  

train_patches, train_labels = extract_patches_from_dataset(train_dataset)
test_patches, test_labels = extract_patches_from_dataset(test_dataset)
train_data = TensorDataset(train_patches, train_labels)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_data = TensorDataset(test_patches, test_labels)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
print(f"Train patches shape: {train_patches.shape}")
print(f"Test patches shape: {test_patches.shape}")
print("done patch extraction")

#above code for patch extraction