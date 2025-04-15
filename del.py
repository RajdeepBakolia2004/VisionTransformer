import numpy
def extract_patches(images, patch_size=2):
    """
    Splits images into non-overlapping patches of given size.
    
    Args:
        images: NumPy array of shape (N, 32, 32, 3) [Batch of images]
        patch_size: Size of each square patch (default 2x2)
    
    Returns:
        Patches: NumPy array of shape (N, num_patches, patch_size, patch_size, 3)
    """
    N,H, W, C = images.shape  # Batch, Height, Width, Channels
    num_patches = (H // patch_size) * (W // patch_size)  # Total patches per image
    
    # Reshape into patches
    patches = images.reshape(N, H//patch_size, patch_size, W//patch_size, patch_size, C)
    
    # Rearrange axes to group patches together
    patches = patches.transpose(0, 1, 3, 2, 4, 5).reshape(N, num_patches, patch_size, patch_size, C)
    return patches

image = numpy.random.randn(1,4,4,2)
original = image.transpose(0, 3, 1, 2)
patches = extract_patches(image, patch_size=2)
print(f"Original Image Shape: {image.shape}")
print(f"Patches Shape: {patches.shape}")
print("original image: ", original)
print("patches: ", patches)