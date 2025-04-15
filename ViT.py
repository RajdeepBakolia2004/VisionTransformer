import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        # Combine all Q, K, V projections in one linear layer for efficiency
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, N, D = x.shape  # [batch, tokens, dim]
        qkv = self.qkv_proj(x)  # [B, N, 3 * dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]

        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B, heads, N, head_dim]

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
        attn = torch.softmax(scores, dim=-1)

        # Weighted sum of values
        out = torch.matmul(attn, v)  # [B, heads, N, head_dim]
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)  # [B, N, dim]

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

        # Learnable components
        self.linear_proj = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # Transformer blocks
        self.transformer = nn.Sequential(*[
            TransformerBlock(dim, heads, mlp_dim)
            for _ in range(depth)
        ])

        
        self.to_cls = nn.Linear(dim, dim)  
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        B, N, _ = x.shape
        x = self.linear_proj(x)  # [B, N, D]
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)          # [B, N+1, D]
        x = x + self.pos_embed                       

        x = self.transformer(x)                        

        cls_output = x[:, 0]                           
        return self.mlp_head(self.to_cls(cls_output))  