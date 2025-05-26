'''import torch.nn as nn

class StudentWithUncertainty(nn.Module):
    def __init__(self):
        super().__init__()
        # Implement student architecture with uncertainty heads
        self.feature_extractor = nn.Sequential(
            # Your implementation here
        )
        self.mean_head = nn.Linear(feature_dim, output_dim)
        self.logvar_head = nn.Linear(feature_dim, output_dim)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.mean_head(features), self.logvar_head(features)'''
import timm
import torch
import torch.nn as nn
import copy  
import torch
import torch.nn as nn
from .components.patch_embed import PatchEmbed
class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed(image_size=224, patch_size=16, embed_dim=192)
        
        # 2. Class Token + Positional Embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, 192))
        num_patches = (224 // 16) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, 192))
        
        # 3. Transformer Layers
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=192,
                nhead=8,
                batch_first=True  # Fixes nested tensor warning
            ),
            num_layers=4
        )
        
        # 4. MC-Dropout
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # Patch embeddings
        x = self.patch_embed(x)  # [B, 196, 192]
        
        # Add class token
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # [B, 197, 192]
        x += self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # MC-Dropout (uncertainty sampling)
        if not self.training:
            return torch.stack([self.dropout(x) for _ in range(5)])
        return x
'''
class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, image_size=224, patch_size=16, embed_dim=192):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels=3,  # CIFAR-100 has 3 channels
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, x):
        # x: [B, 3, 224, 224]
        x = self.proj(x)  # [B, embed_dim, 14, 14] (for patch_size=16)
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x
class Student(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Patch Embedding Layer
        self.patch_embed = PatchEmbed(
            image_size=224,
            patch_size=16,
            embed_dim=192
        )
        
        # 2. Class Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, 192))
        
        # 3. Positional Embeddings
        num_patches = (224 // 16) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, 192))
        
        # 4. Transformer Encoder (Simplified example)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=192, nhead=8),
            num_layers=4
        )
        
        # 5. MC-Dropout
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # 1. Patch Embedding
        x = self.patch_embed(x)  # [B, num_patches, 192]
        
        # 2. Add Class Token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, 192]
        
        # 3. Add Positional Embeddings
        x += self.pos_embed
        
        # 4. Transformer Encoder
        x = self.transformer(x)
        
        # 5. MC-Dropout (for uncertainty)
        if self.training:
            return x
        else:
            # Return multiple samples with dropout
            return torch.stack([self.dropout(x) for _ in range(5)])
            '''