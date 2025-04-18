import torch
from torchvision import transforms
import torch.nn as nn 
from PIL import Image


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
    def __init__(self, dim, heads=4, mlp_dim=128):
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
    
class SimpleViT(nn.Module):
    def __init__(self, patch_dim, num_patches, dim, depth=6, heads=4, mlp_dim=128, num_classes=10):
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

        
        self.to_cls = nn.Linear(dim, dim)  
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


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def patching(X):
    patch_size = 4
    stride = 4
    X = X.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    X = X.permute(0, 2, 3, 1, 4, 5)
    X = X.reshape(X.shape[0] ,-1, patch_size * patch_size )
    return X


model = SimpleViT(patch_dim=16, num_patches=49, dim=64).to(device)
checkpoint_path = f'mnistvit.pth'
model.load_state_dict(torch.load(checkpoint_path))



model.eval()
for i in range(6, 11):
    img_path = f"/home/rajdeep/final/{i}.png" # Replace with actual image path
    image = Image.open(img_path).convert("L")
    image_tensor = transform(image).unsqueeze(0).to(device) 

    patches = patching(image_tensor) 
    with torch.no_grad():
        output = model(patches)
        predicted = output.argmax(dim=1).item()
        print(f"Predicted class for image {i}: {predicted}")
