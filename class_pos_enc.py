import numpy as np
from embedding import patch_embedding
def class_embedding(patch_embeddings = patch_embedding(), class_token=None):  
    """
    Adds a learnable class token to the patch embeddings.

    Args:
        patch_embeddings: NumPy array of shape (N, num_patches, D)

    Returns:
        class_embedded: NumPy array of shape (N, num_patches + 1, D)
    """
    N, num_patches, D = patch_embeddings.shape

    if class_token is None:
        # Initialize a learnable class token of shape (1, D)
        class_token = np.random.randn(1, D)

    # Repeat class token for each image in batch (N, 1, D)
    class_tokens = np.tile(class_token, (N, 1, 1))

    # Prepend class token to patch embeddings → (N, num_patches + 1, D)
    class_embedded = np.concatenate([class_tokens, patch_embeddings], axis=1)
    return class_embedded


def add_positional_encoding(class_embedded = patch_embedding()):
    """
    Adds positional encoding to the class-embedded vectors.

    Args:
        class_embedded: NumPy array of shape (N, num_patches + 1, D)

    Returns:
        pos_encoded: NumPy array of shape (N, num_patches + 1, D)
    """
    N, num_patches_plus_one, D = class_embedded.shape


    #add sine and cosine positional encoding
    # Initialize positional encoding matrix
    pos_encoding = np.zeros((num_patches_plus_one, D))

    # Compute positional encodings for each position
    for pos in range(num_patches_plus_one):
        for i in range(0, D, 2):
            pos_encoding[pos, i] = np.sin(pos / 10000 ** ((2 * i) / D))
            pos_encoding[pos, i + 1] = np.cos(pos / 10000 ** ((2 * i) / D))


    # Add positional encoding to class-embedded vectors
    pos_encoded = class_embedded + pos_encoding

    return pos_encoded

if __name__ == "__main__":
    patch_embeddings = patch_embedding()


    # Step 1: Add class token
    class_embedded = class_embedding(patch_embeddings)
    print(f"After Class Token: {class_embedded.shape}")  # (50000, 65, 128)

    # Step 2: Add positional encoding
    pos_encoded = add_positional_encoding(class_embedded)
    print(f"After Positional Encoding: {pos_encoded.shape}")  # (50000, 65, 128)
