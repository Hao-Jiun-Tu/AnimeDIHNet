import torch
import torch.nn as nn

class UNetEncoder(nn.Module):
    def __init__(self):
        super(UNetEncoder, self).__init__()
        
        self.conv_in = nn.Sequential(
            ConvBlock(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1, activation=nn.ReLU, bias=True),
            ConvBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, activation=nn.ReLU, bias=True)
        )

        block0 = UNetDownBlock(in_channels=64, out_channels=128, activation=nn.ReLU, padding=1)
        block1 = UNetDownBlock(in_channels=128, out_channels=128, activation=nn.ReLU, padding=1)
    
        block2 = UNetDownBlock(in_channels=128, out_channels=256, activation=nn.ReLU, padding=1)
        block3 = UNetDownBlock(in_channels=256, out_channels=256, activation=nn.ReLU, padding=1)
    
        block4 = UNetDownBlock(in_channels=256, out_channels=512, activation=nn.ReLU, padding=1)
        self.down_blocks = nn.Sequential(
            block0, block1, block2, block3, block4
        )
        
    def forward(self, x):
        outputs = []
        output = self.conv_in(x)
        outputs.append(output)
        
        for down_block in self.down_blocks:
            output = down_block(output)
            outputs.append(output)
            # print('down_block shape: {}'.format(output.shape))
            
        return outputs[::-1]


class UNetDecoder(nn.Module):
    def __init__(self):
        super(UNetDecoder, self).__init__()

        block0 = UNetUpBlock(in_channels=512, out_channels=256, activation=nn.ReLU, padding=1)
        block1 = UNetUpBlock(in_channels=256, out_channels=256, activation=nn.ReLU, padding=1)
    
        block2 = UNetUpBlock(in_channels=256, out_channels=128, activation=nn.ReLU, padding=1)
        block3 = UNetUpBlock(in_channels=128, out_channels=128, activation=nn.ReLU, padding=1)
    
        block4 = UNetUpBlock(in_channels=128, out_channels=64, activation=nn.ReLU, padding=1)
        block5 = UNetUpBlock(in_channels=64, out_channels=64, activation=nn.ReLU, padding=1)
        
        self.up_blocks = nn.Sequential(
            block0, block1, block2, block3, block4, block5
        )
        
        self.conv_out = nn.Sequential(
            ConvBlock(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, activation=nn.ReLU, bias=True)
        )

    def forward(self, encoder_outputs):
        output = encoder_outputs[0]
        for up_block, skip_output in zip(self.up_blocks, encoder_outputs[1:]):
            output = up_block(output) + skip_output
            # print('up_block shape: {}'.format(up_block(output).shape))
            # print('skip_out shape: {}'.format(skip_output.shape))

        output = self.conv_out(output)
        return output


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation, padding):
        super(UNetDownBlock, self).__init__()
        self.convs = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=padding, activation=activation, bias=True),
            ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=padding, activation=activation, bias=True)
        )
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2) #if pool else nn.Identity()

    def forward(self, x):
        output = self.convs(x)
        output = self.pooling(output)
        return output
    
    
class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation, padding):
        super(UNetUpBlock, self).__init__()
        self.deconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True),
            ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=padding, activation=activation, bias=True),
            ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=padding, activation=activation, bias=True),
        )      
        
        # self.deconv = nn.Sequential(
        #     ConvBlock(in_channels, 4*in_channels, kernel_size=3, stride=1, padding=padding, activation=activation, bias=True),
        #     nn.PixelShuffle(2),
        #     ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=padding, activation=activation, bias=True),
        #     ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=padding, activation=activation, bias=True)
        # )      

    def forward(self, x):
        output = self.deconv(x)
        return output   


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=nn.ReLU, bias=True):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            activation(),
        )

    def forward(self, x):
        return self.block(x)