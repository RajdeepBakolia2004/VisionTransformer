from patch import patch_image
import numpy as np


def patch_embedding(patches = patch_image(4), D = 128, embedding = None):   
    """
    Converts 4x4 patches into D-dimensional embeddings using linear projection.
    
    Args:
        patches: NumPy array of shape (N, num_patches, 4, 4, 3) → (Batch, Patches, H, W, C)
        D: Output embedding dimension

    Returns:
        embeddings: NumPy array of shape (N, num_patches, D)
    """
    N, num_patches, H, W, C = patches.shape  # Extract shape
    
    # Flatten each patch to (N, num_patches, H*W*C)
    flattened_patches = patches.reshape(N, num_patches, -1)  # Shape: (N, num_patches, 48)
    
    if embedding is not None:
        # Use pre-defined embedding matrix
        W = embedding
    else:
        # Initialize random weight matrix W for linear projection (48 → D)
        W = np.random.randn(48, D)  # Shape: (48, D)
    
    # Perform linear projection: (N, num_patches, 48) @ (48, D) → (N, num_patches, D)
    embeddings = flattened_patches @ W  
    return embeddings




if __name__ == "__main__":
    D = 128  # Embedding dimension
    patches = patch_image(4)
    embeddings = patch_embedding(patches, D)
    print(f"Patches Shape: {patches.shape}")  # (50000, 64, 4, 4, 3)
    print(f"Embeddings Shape: {embeddings.shape}")  # (50000, 64, 128)
