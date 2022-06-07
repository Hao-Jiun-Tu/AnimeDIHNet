import torch
import torch.nn as nn
from unet import UNetEncoder, UNetDecoder

class AnimeDIHNet(nn.Module):
    def __init__(self):
        super(AnimeDIHNet, self).__init__()
        self.autoencoder = nn.Sequential(
            UNetEncoder(),
            UNetDecoder()
        )
        
        # self.en = UNetEncoder()
        # self.de = UNetDecoder()

    def forward(self, x):
        # print('Input shape: {}'.format(x.shape))
        
        # output = self.en(x)
        # print('UnetEncoder shape: {}'.format(output.shape))
        # output = self.de(output)
        # print('UnetDecoder shape: {}'.format(output.shape))
        
        output = self.autoencoder(x) 
        return output