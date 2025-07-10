import torch
import torch.nn as nn
from torch.nn import functional as F

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class ConvBNMish(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()
    
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class TransitionRegion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBNMish(channels, channels, 1)
        self.conv2 = ConvBNMish(channels, channels, 3)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        return x1 + x2

class C3TR(nn.Module):
    def __init__(self, in_channels, out_channels, n=1):
        super().__init__()
        mid_channels = out_channels // 2
        self.conv1 = ConvBNMish(in_channels, mid_channels, 1)
        self.conv2 = ConvBNMish(in_channels, mid_channels, 1)
        
        self.tr_blocks = nn.Sequential(*[TransitionRegion(mid_channels) for _ in range(n)])
        
        self.conv3 = ConvBNMish(mid_channels, mid_channels, 1)
        self.conv_out = ConvBNMish(out_channels, out_channels, 1)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.tr_blocks(x2)
        x2 = self.conv3(x2)
        x = torch.cat([x1, x2], dim=1)
        return self.conv_out(x)