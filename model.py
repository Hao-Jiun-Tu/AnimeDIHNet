import torch
import torch.nn as nn
from unet import UNetEncoder, UNetDecoder

class AnimeDIHNet(nn.Module):
    def __init__(self):
        super(AnimeDIHNet, self).__init__()
        self.encoder = UNetEncoder()
        self.decoder = UNetDecoder()
        #self.conv1x1_32to3ch = ConvBlock(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=0, activation=nn.ReLU, bias=True)
        #self.conv1x1_32to1ch = ConvBlock(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1, activation=nn.ReLU, bias=True)
        #self.conv1x1_32to1ch = nn.Conv2d(32, 1, 1, stride=1, padding=0)


    def forward(self, x):
        #===== Check Input Dimension =====#
        comp = x[:,:3]
        mask = x[:,3].unsqueeze(1)
        
        #===== AutoEncoder Structure =====#
        out = self.encoder(x) 
        out = self.decoder(out) 
        
        #===== Create Blending Layer =====#
        # out_comp = self.conv1x1_32to3ch(out)
        # out_mask = torch.sigmoid(3.0 *self.conv1x1_32to1ch(out))
        # output = out_comp*(1 - out_mask) + out_mask*comp
        
        return out
    
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, activation=nn.ReLU, bias=True):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            activation(),
        )

    def forward(self, x):
        return self.block(x)