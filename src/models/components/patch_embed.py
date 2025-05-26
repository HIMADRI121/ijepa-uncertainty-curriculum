import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, image_size=224, patch_size=16, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, 
                            kernel_size=patch_size, 
                            stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x