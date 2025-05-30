from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,TensorDataset
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim



DATASET_PATH = "cifar10_images"
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_dataset = ImageFolder(root="cifar10_images/test", transform=transform)



def extract_patches(img_tensor, P):
    C, H, W = img_tensor.shape
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


# ----------------------------
# Patch prep and dataloader
# ----------------------------
patch_size = 4
test_patches, test_labels = transform_dataset_to_patches(test_dataset, patch_size)
test_data = TensorDataset(test_patches, test_labels)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)   




class SimpleViT(nn.Module):
    def __init__(self, patch_dim, num_patches, dim, depth=6, heads=4, mlp_dim=256, num_classes=10):
        super(SimpleViT, self).__init__()
        self.dim = dim
        self.num_patches = num_patches
        self.patch_dim = patch_dim
        self.linear_proj = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.transformer = nn.Sequential(*[
            TransformerBlock(dim, heads, mlp_dim)
            for _ in range(depth)
        ])
        self.to_cls = nn.Identity()  
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        B, N, _ = x.shape
        x = self.linear_proj(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)          
        x = x + self.pos_embed                     
        x = self.transformer(x)                    
        cls_output = x[:, 0]                           
        return self.mlp_head(self.to_cls(cls_output))  



class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.scale = self.head_dim ** -0.5
    def forward(self, x):
        B, N, D = x.shape  
        qkv = self.qkv_proj(x)  
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) 
        q, k, v = qkv[0], qkv[1], qkv[2] 
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale 
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v) 
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)
        return self.out_proj(out)
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_dim=256):
        super().__init__()
        self.attn = MultiHeadAttention(dim, heads)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))  
        x = x + self.mlp(self.ln2(x))
        return x
    

model = SimpleViT(patch_dim=48, num_patches=64, dim=128, depth=6, heads=4, mlp_dim=256, num_classes=10)
model.load_state_dict(torch.load("vit_model.pth"))
model.eval()

correct = 0
total = 0
for img, label in test_loader:
    with torch.no_grad():
        img = img.view(img.size(0), -1, 48)
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        print(f'Predicted: {predicted}, Actual: {label}')
print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')

