import torch
import torch.nn as nn
from unet_att import UNetEncoder, UNetDecoder

class AnimeDIHNet(nn.Module):
    def __init__(self):
        super(AnimeDIHNet, self).__init__()
        self.encoder = UNetEncoder()
        self.decoder = UNetDecoder()
        self.conv3x3_32to3ch = ConvBlock(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1, activation=nn.ReLU, bias=True)

    def forward(self, x):
        #===== Check Input Dimension =====#
        comp = x[:,:3]
        mask = x[:,3].unsqueeze(1)
        
        #===== AutoEncoder Structure =====#
        out = self.encoder(x) 
        out = self.decoder(out, mask) 
        
        img = self.conv3x3_32to3ch(out)        
        out = comp*(1.0-mask) + img*mask
        return out
    
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=nn.ReLU, bias=True):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            activation(),
        )

    def forward(self, x):
        return self.block(x)