import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from blocks import ConvBlock, LinearAttentionBlock,GridAttentionBlock, ProjectorBlock
from initialize import *

class VGG(nn.Module):
    def __init__(self, im_size, num_classes, init='xavierUniform'):
        super(VGG, self).__init__()
        # conv blocks
        self.conv_block1 = ConvBlock(3, 64, 2)
        self.conv_block2 = ConvBlock(64, 128, 2)
        self.conv_block3 = ConvBlock(128, 256, 3)
        self.conv_block4 = ConvBlock(256, 512, 3)
        self.conv_block5 = ConvBlock(512, 512, 3)
        self.conv_block5 = ConvBlock(512, 512, 3)
        self.conv_block5 = ConvBlock(512, 512, 3)
        self.conv_block6 = ConvBlock(512, 512, 3)
        self.conv_block7 = ConvBlock(512, 512, 3)
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.classify = nn.Linear(in_=512, out_=3, bias=True)

        # initialize
        if init == 'kaimingNormal':
            weights_init_kaimingNormal(self)
        elif init == 'kaimingUniform':
            weights_init_kaimingUniform(self)
        elif init == 'xavierNormal':
            weights_init_xavierNormal(self)
        elif init == 'xavierUniform':
            weights_init_xavierUniform(self)
        else:
            raise NotImplementedError("Invalid type of initialization!")
    def forward(self, x):
        # feed forward
        x = self.conv_block1(x)    #512,64
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)    #256,64
        x = self.conv_block2(x)                         #256,128
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)        #128,128
        x = self.conv_block3(x)                #128,256
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)    #64,256
        x = self.conv_block4(x)  # 64,512
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)  # 32,512
        l1 = self.conv_block5(x)  # 32,512
        x = F.max_pool2d(l1, kernel_size=2, stride=2, padding=0)  # 16,512
        l2 = self.conv_block6(x)  # 16,512
        x = F.max_pool2d(l2, kernel_size=2, stride=2, padding=0)  # 8,512
        l3 = self.conv_block6(x)  # 8,512
        g = self.conv_block7(x)  # 8,512



        g=self.avgpool(g)
        g = g.squeeze(dim=-1)
        g = g.squeeze(dim=-1)


        c1, c2, c3 = None, None, None


        x = self.classify(g)

        return [x, c1, c2, c3]



