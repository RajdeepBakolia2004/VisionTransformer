from torchvision import transforms
import torch.nn as nn
import torch
import torch
from torchvision import transforms
from PIL import Image




class Patch_Linear_Embedding(nn.Module):
    def __init__(self, patch_size=4, embedding_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        self.proj = nn.Linear(patch_size * patch_size, embedding_dim)


    def forward(self, x):
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5)
        x = x.reshape(x.shape[0] ,x.shape[1],x.shape[2] , self.patch_size * self.patch_size )
        x = self.proj(x)
        return x


class W_MSA(nn.Module):
    def __init__(self, dim, window_size=6, num_heads=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        """
        x: (B, H, W, D)
        """
        B, H, W, D = x.shape
        ws = self.window_size
        x = x.reshape(B, H // ws, ws, W // ws, ws, D)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(-1, ws * ws, D)


        M, V, C = x.shape  
        qkv = self.qkv(x) 
        qkv = qkv.reshape(M, V, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  

        q, k, v = qkv[0], qkv[1], qkv[2]  

        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  
        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)  
        out = out.permute(0, 2, 1, 3).reshape(M, V, C)  

        out = out.reshape(B, H // ws, W // ws, ws, ws, D)  
        out = out.permute(0, 1, 3, 2, 4, 5) 
        out = out.reshape(B, H, W, D)  
        return out


def mask(B, H, W,ss,ws):
    mask = torch.zeros(B, H, W, dtype=torch.int)
    mask[:, :ss, :ss] = 0
    mask[:, :ss, ss:] = 1
    mask[:, ss:, :ss] = 2
    mask[:, ss:, ss:] = 3
    shift_size = ss
    shifted_mask = torch.roll(mask, shifts=(-shift_size, -shift_size), dims=(1, 2))
    shifted_mask = shifted_mask.reshape(B, H // ws, ws, W // ws, ws)
    shifted_mask = shifted_mask.permute(0, 1, 3, 2, 4)
    shifted_mask = shifted_mask.reshape(-1, ws * ws)
    N, L = shifted_mask.shape
    eq = shifted_mask.unsqueeze(2) == shifted_mask.unsqueeze(1)
    attn_mask = torch.where(eq, torch.tensor(0.0), torch.tensor(float('-inf')))
    return attn_mask


class SW_MSA(nn.Module):
    def __init__(self, dim, window_size=6, shift_size=3, num_heads=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: (B, H, W, D)
        """
        B, H, W, D = x.shape
        ws = self.window_size
        ss = self.shift_size
        x = torch.roll(x, shifts=(-ss, -ss), dims=(1, 2))


        attn_mask = mask(B, H, W, ss, ws)
        attn_mask = mask(B, H, W, ss, ws).to(x.device)



        x = x.reshape(B, H // ws, ws, W // ws, ws, D)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(-1, ws * ws, D)

        M, V, C = x.shape  
        qkv = self.qkv(x)  
        qkv = qkv.reshape(M, V, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  

        q, k, v = qkv[0], qkv[1], qkv[2]  

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  
        scores = scores + attn_mask.unsqueeze(1)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(M, V, C)

        out = out.reshape(B, H // ws, W // ws, ws, ws, D)
        out = out.permute(0, 1, 3, 2, 4, 5)
        out = out.reshape(B, H, W, D)
        out = torch.roll(out, shifts=(ss, ss), dims=(1, 2))
        out = self.proj(out)
        return out
    

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads = 8, window_size=6, shift_size=3):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = W_MSA(dim, num_heads=num_heads, window_size=window_size)

        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.attn2 = SW_MSA(dim, num_heads=num_heads, window_size=window_size, shift_size=shift_size)


        self.mlp2 = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )



    def forward(self, x):
        B, H, W, C = x.shape

        x = x + self.attn1(self.norm1(x))
        x = x + self.mlp1(self.norm2(x))
        x = x + self.attn2(self.norm3(x))
        x = x + self.mlp2(self.norm4(x))
        return x
    

class PatchMerging(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.reduction = nn.Linear(4 * input_dim, 2 * input_dim, bias=False)
        self.norm = nn.LayerNorm(4 * input_dim)

    def forward(self, x):
        B, H, W, C = x.shape

        x0 = x[:, 0::2, 0::2, :]  
        x1 = x[:, 0::2, 1::2, :]  
        x2 = x[:, 1::2, 0::2, :]  
        x3 = x[:, 1::2, 1::2, :]  
        x_merged = torch.cat([x0, x1, x2, x3], dim=-1) 
        x_merged = self.norm(x_merged)
        x_merged = self.reduction(x_merged)  
        return x_merged

class ClassificationHead(nn.Module):
    def __init__(self, embed_dim=384, num_classes=10):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.avg_pool(x)           
        x = x.reshape(x.size(0), -1)     
        return self.fc(x)              


class SimpleSwinTransformer(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.embedding = Patch_Linear_Embedding(patch_size=4, embedding_dim=96)
        self.stb1 = SwinTransformerBlock(dim=96, window_size=6, shift_size=3, num_heads=8)
        self.stb2 = SwinTransformerBlock(dim=96, window_size=6, shift_size=3, num_heads=8)
        self.stb3 = SwinTransformerBlock(dim=192, window_size=6, shift_size=3, num_heads=8)
        self.stb4 = SwinTransformerBlock(dim=192, window_size=6, shift_size=3, num_heads=8)
        self.stb5 = W_MSA(dim = 384, window_size=6, num_heads=8)
        self.stb6 = W_MSA(dim = 384, window_size=6, num_heads=8)
        self.patch_merging1 = PatchMerging(input_dim=96)
        self.patch_merging2 = PatchMerging(input_dim=192)
        self.classification_head = ClassificationHead(embed_dim=384, num_classes=num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.stb1(x)
        x = self.stb2(x)
        x = self.patch_merging1(x)
        x = self.stb3(x)
        x = self.stb4(x)
        x = self.patch_merging2(x)
        x = self.stb5(x)
        x = self.stb6(x)
        x = self.classification_head(x)
        return x




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleSwinTransformer(num_classes=10).to(device)  
model.load_state_dict(torch.load("mnistswin.pth", map_location=device), strict=False)
model.eval()

transform = transforms.Compose([
    transforms.Resize((96, 96)),             
    transforms.Grayscale(num_output_channels=1),  
    transforms.ToTensor(),                  
    transforms.Normalize((0.5,), (0.5,))      
])

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


for i in range(6, 11):
    img_path = f"/home/rajdeep/final/{i}.png" # Replace with actual image path
    image = Image.open(img_path).convert("L")  
    input_tensor = transform(image).unsqueeze(0).to(device)  
    

    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()

    print(f"Image {i}.png → Predicted Class: {classes[pred_class]} ({pred_class})")
