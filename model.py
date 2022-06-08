import torch
import torch.nn as nn
from unet import UNetEncoder, UNetDecoder

class AnimeDIHNet(nn.Module):
    def __init__(self):
        super(AnimeDIHNet, self).__init__()
        self.encoder = UNetEncoder()
        self.decoder = UNetDecoder()
        self.conv1x1_32to3ch = ConvBlock(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=0, activation=nn.ReLU, bias=True)
        self.conv1x1_32to1ch = ConvBlock(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, activation=nn.ReLU, bias=True)

    def forward(self, x):
        #===== Check Input Dimension =====#
        # print('Input shape: {}'.format(x.shape))
        comp = x[:,:3]
        mask = x[:,3].unsqueeze(1)
        # print('mask shape: {}'.format(mask.shape))
        # print('comp shape: {}'.format(comp.shape))
        
        #===== AutoEncoder Structure =====#
        out = self.encoder(x) 
        out = self.decoder(out) 
        
        #===== Create Blending Layer =====#
        out_comp = self.conv1x1_32to3ch(out)
        out_mask = self.conv1x1_32to1ch(out)
        output = (out_comp*out_mask) + (1-out_mask)*comp
        
        return output
    
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, activation=nn.ReLU, bias=True):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            activation(),
        )

    def forward(self, x):
        return self.block(x)