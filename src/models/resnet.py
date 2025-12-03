import torch
import torch.nn as nn
import torch.nn.functional as F

def exists(x):
    return x is not None


class RMSNorm(nn.Module):
    """
    RMS Normalization
    F.normalizae
    """
    def __init__(self, dim):
        super(RMSNorm, self).__init__()
        
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1,1))
        
    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * self.scale
    

class ResNetBlock(nn.Module):
    expansion = 1
    def __init__(self, cin, cout, *, time_emb_dim=None, dropout=0.):
        super(ResNetBlock, self).__init__()
        self.expansion = ResNetBlock.expansion
        
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, cout *2),
        ) if exists(time_emb_dim) else None 
        
        self.conv1 = nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False)
        self.norm1 = RMSNorm(cout)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False)
        self.norm2 = RMSNorm(cout)
        self.shortcut = nn.Conv2d(cin, cout, kernel_size=1, bias=False) if  cin != cout else nn.Identity()
        
    def forward(self,x, time_emb=None):
        out = self.norm1(self.conv1(x))
        
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
            scale, shift = time_emb.chunk(2, dim=1)
            out = out * (scale +1) + shift   
        
        out = F.silu(out)    
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        out = F.silu(out)
        out = self.dropout(out)
        out += self.shortcut(x)
        return out


if __name__ == "__main__":
    block = ResNetBlock(3, 64)
    print(block)