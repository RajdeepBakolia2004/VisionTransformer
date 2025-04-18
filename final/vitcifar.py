from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch
from torchvision import transforms
from PIL import Image


def patch(image,patch_size, stride):
    channel = image.shape[1]
    image = image.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    image = image.permute(0, 2, 3, 1, 4, 5)
    image = image.reshape(image.shape[0], -1, patch_size * patch_size * channel)
    return image

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
    def __init__(self, dim, heads=8, mlp_dim=384):
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
    def __init__(self, patch_dim, num_patches, dim, depth=8, heads=8, mlp_dim=384, num_classes=10):
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
    



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleViT(48,64,128).to(device)
model.load_state_dict(torch.load("cifarvit.pth", map_location=device), strict=False)
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),         
    transforms.ToTensor(),               
    transforms.Normalize((0.5, 0.5, 0.5), 
                         (0.5, 0.5, 0.5))
])


classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']



for i in range(1, 6):
    img_path = f"/home/rajdeep/final/{i}.png"  # Replace with actual image path
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device) 
    input_tensor = patch(input_tensor, 4, 4)  

    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
    
    print(f"Image {i}.png → Predicted Class: {classes[pred_class]} ({pred_class})")
