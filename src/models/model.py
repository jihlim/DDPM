import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import exists, RMSNorm, ResNetBlock

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length=1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)


class Downsample(nn.Module):
    def __init__(self, cin, cout=None):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(cin * 4, default(cout, cin), kernel_size=1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.contiguous().view(b, c*4, h//2, w//2)
        out = self.conv(x)
        return out


class Upsample(nn.Module):
    def __init__(self, cin, cout=None):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(cin, default(cout, cin), kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.conv(x)
        return out
    

class SimpleUNet(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super(SimpleUNet, self).__init__()
        
        self.init_conv = nn.Conv2d(3, 128, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        
        # Encoder
        self.enc1 = ResNetBlock(128, 128, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.enc2 = ResNetBlock(128, 256, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.enc3 = ResNetBlock(256, 512, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(512)
        
        # Middle
        self.mid1 = ResNetBlock(512, 512, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.mid2 = ResNetBlock(512, 512, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        # Decoder
        self.dec3 = nn.ConvTranspose2d(512, 256, kernel_size=3, strdie=2, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.dec2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.dec2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(64)
        
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(3)

    def forward(self, x, t):
        x = F.LayerNorm(self.bn0(self.init_conv(x)))
        
        return x
    

class SelfAttention(nn.Module):
    def __init__(self, dim, heads=4, d_head=32,):
        super(SelfAttention, self).__init__()
        
        self.heads = heads
        self.d_head = d_head
        self.scale = d_head ** 0.5
        hidden_dim = d_head * heads 
        
        self.to_qkv = nn.Conv2d(dim, hidden_dim*3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, kernel_size=1) 
    
    def forward (self, x):
        b, c, h, w = x.shape
        
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q = q.contiguous().view(b * self.heads, -1, h * w).transpose(1, 2)                      # size = [batch_size * heads, h*w, d_heads]
        k = k.contiguous().view(b * self.heads, -1, h * w).transpose(1, 2)                      # size = [batch_size * heads, h*w, d_heads]
        v = v.contiguous().view(b * self.heads, -1, h * w).transpose(1, 2)                      # size = [batch_size * heads, h*w, d_heads]
        
        q = q / self.scale
        
        attn_weight = torch.bmm(q, k.transpose(1,2))                                            # size = [batch_size * heads, h*w, h*w]
        attn_weight = F.softmax(attn_weight, dim=-1)
        
        out = torch.bmm(attn_weight, v)                                                         # size = [batch_size * heads, h*w, d_heads]
        out = out.transpose(1,2).contiguous().view(b, self.heads * self.d_head, h, w)
        out = self.to_out(out)
        return out


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, d_head=32,):
        super(LinearAttention, self).__init__()
        
        self.heads = heads
        self.d_head = d_head
        self.scale = d_head ** 0.5
        hidden_dim = d_head * heads 
        
        self.to_qkv = nn.Conv2d(dim, hidden_dim*3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, kernel_size=1) 
    
    def forward (self, x):
        b, c, h, w = x.shape
        
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q = q.contiguous().view(b * self.heads, -1, h * w)                                      # size = [batch_size * heads, d_heads, h*w]
        k = k.contiguous().view(b * self.heads, -1, h * w)                                      # size = [batch_size * heads, d_heads, h*w]
        v = v.contiguous().view(b * self.heads, -1, h * w)                                      # size = [batch_size * heads, d_heads, h*w]
        
        q.softmax(dim =-2)
        k.softmax (dim=-1)
        
        q = q / self.scale
        
        context = torch.bmm(k, v.transpose(1,2))                                            # size = [batch_size * heads, d_heads, d_heads]
        
        out = torch.bmm(context.permute(1,2), q)                                            # size = [batch_size * heads, d_heads, h*w]
        out = out.contiguous().view(b, self.heads * self.d_head, h, w)                      # size = [batch_size,  heads * d_heads, h, w]
        out = self.to_out(out)
        return out


class TimeEmbedding(nn.Module):
    """
    Time Embedding for DDPM
    """
    def __init__(self, dim, theta=10000.0):
        super (TimeEmbedding, self).__init__()
        self.dim = dim
        self.theta = theta
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim -1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim =-1)                                    # emb = [[sin(), cos()], [sin(), cos()], ... ,[sin(), cos()]]
        return emb


class PositionalEncodings(nn.Module):
    """
    Positional Encodings for Transformer (Attention is All You Need)
    """
    def __init__(self, d_model=512, max_seq_length=1000):
        super(PositionalEncodings, self).__init__()
        
        self.pe = torch.zeros(max_seq_length, d_model)
        dimension = torch.arange(0, d_model)
        d_even = dimension[::2]
        d_odd = dimension[1::2]
        position = torch.arange(0, max_seq_length)
        for pos in position:
            self.pe[pos][::2] = torch.sin(pos.float()/(10000.0 ** (d_even/d_model)))
            self.pe[pos][1::2] = torch.cos(pos.float()/(10000.0 ** (d_odd/d_model)))
    
    def forward(self,x):
        return x + self.pe


class UNet(nn.Module):
    def __init__(
        self,
        dim: int,                                                               # channel dimension after stem layer(initial convolution)
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        time_pos_emb_theta = 10000.0,
        dropout = 0,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,
    ):
        super(UNet, self).__init__()

        self.channels = channels                                                # input image channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)
        
        # Initial of the Model 
        init_dim = default(init_dim, dim)                                       # cout after Stem Layer
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)      # Stem Layer
        
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]                   # dims = [(init_)dim, dim*1, dim*2, dim*4, dim*8]
        in_out = list(zip(dims[:-1], dims[1:]))                                 # list of tuples: (cin, cout)
        
        # Time Embedding
        time_dim = dim * 4                                                      # time dimension = dim * 4
    
        time_pos_emb = TimeEmbedding(dim, time_pos_emb_theta)
    
        self.time_mlp = nn.Sequential(
            time_pos_emb,
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Attention
        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)                          # (False, False, False, True)

        num_stages = len(dim_mults)                                                         # Number of stages
        full_attn  = cast_tuple(full_attn, num_stages)                                      # (False, False, False, True) or (True, True, True, True)
        attn_heads = cast_tuple(attn_heads, num_stages)                                     # (attn_heads, attn_heads, attn_head, attn_heads) == (4,4,4,4)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)                               # (attn_dim_head, attn_dim_head, attn_dim_head, attn_dim_head) == (32, 32, 32, 32)
        
        # Blocks
        FullAttention = SelfAttention
        resnet_block = partial(ResNetBlock, time_emb_dim= time_dim, dropout=dropout)
        
        
        # Layers
        self.downs = nn.ModuleList([])                                                      # List of module lists
        self.ups  = nn.ModuleList([])                                                       # List of module lists
        num_resolutions = len(in_out)                                                       # number of layers
        
        # Downsampling
        for idx, ((d_in, d_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = idx >= (num_resolutions -1)
            
            attn_block = FullAttention if full_attn else LinearAttention
            
            self.downs.append(nn.ModuleList([
                resnet_block(d_in, d_in),
                resnet_block(d_in, d_in),
                attn_block(d_in),
                Downsample(d_in, d_out) if not is_last else nn.Conv2d(d_in, d_out,  kernel_size=3, padding=1)   
            ]))
        
        # Middle Layers
        d_mid = dims[-1]
        self.mid1 = resnet_block(d_mid, d_mid)
        self.mid_attn = FullAttention(d_mid, )
        self.mid2 = resnet_block(d_mid, d_mid)
        
        # Upsampling
        for idx, ((d_in, d_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = idx >= (num_resolutions -1)
            
            attn_block = FullAttention if full_attn else LinearAttention

            self.ups.append(nn.ModuleList([
                resnet_block(d_out + d_in, d_out),
                resnet_block(d_out + d_in, d_out),
                attn_block(d_out),
                Upsample(d_out, d_in) if not is_last else nn.Conv2d(d_out, d_in, kernel_size=3, padding=1)   
            ]))

        # Final Layer
        default_out_dim = channels                                                      # Input image channels: 3
        self.final_d_out = default(out_dim, default_out_dim)
        
        self.final_res_block = resnet_block(init_dim *2, init_dim)                      # dim * 2 -> dim
        self.final_conv = nn.Conv2d(init_dim, self.final_d_out, kernel_size=1)          # dim -> 3 (input image channels)
    
    
    def forward(self, x, time):   
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim =1)
        
        # Initial  
        x = self.init_conv(x)
        r = x.clone()
        
        t = self.time_mlp(time)
        
        h = []
        
        # Downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            
            x = block2(x, t)
            x = attn(x) + x 
            h.append(x)
            
            x = downsample(x)
        
        # Middle  
        x = self.mid1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid2(x, t)
        
        # Upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x) + x
            
            x = upsample(x)
        
        # Final    
        x = torch.cat((x, r), dim=1)
        
        x = self.final_res_block(x)
        out = self.final_conv(x)
        return out   


if __name__ == "__main__":
    pe = PositionalEncodings()