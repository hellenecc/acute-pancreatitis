import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from blocks import ConvBlock, LinearAttentionBlock,GridAttentionBlock, ProjectorBlock
from initialize import *



class AttnVGG(nn.Module):
    def __init__(self, im_size, num_classes, attention=True, normalize_attn=True, init='xavierUniform'):
        super(AttnVGG, self).__init__()
        self.attention = attention
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
        self.relu = nn.ReLU(inplace=True)




        self.linear1=nn.Linear(in_=1536, out_=30, bias=True)
        #self.linear2 = nn.Linear(in_=1568, out_=400, bias=True)

        # Projectors & Compatibility functions
        if self.attention:
            #self.projector = ProjectorBlock(256, 512)
            self.attn1 = GridAttentionBlock(in__l=512,in__g=512,attn_features=512, up_factor=4,normalize_attn=normalize_attn)
            self.attn2 = GridAttentionBlock(in__l=512,in__g=512,attn_features=512, up_factor=2,normalize_attn=normalize_attn)
            self.attn3 = GridAttentionBlock(in__l=512,in__g=512,attn_features=512, up_factor=1,normalize_attn=normalize_attn)
        # final classification layer
        if self.attention:
            self.classify = nn.Linear(in_=1568, out_=3, bias=True)

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
    def forward(self, inputs_plant, clinic):


        # feed forward
        x = self.conv_block1(inputs_plant)    #512,64
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





        #x2a=self.linear1(x2)
        # pay attention
        if self.attention:
            c1, g1 = self.attn1(l1, g)
            c2, g2 = self.attn2(l2, g)
            c3, g3 = self.attn3(l3, g)
            g = torch.cat((g1,g2,g3), dim=1)

            #g = self.linear1(g)
            #g = self.relu(g)
            #g = self.linear2(g)
            #g = self.relu(g)

            gmerge=torch.cat((g,clinic), dim=1)
            #gmerge=self.linear2(gmerge)
            #gmerge=self.relu(gmerge)


            # batch_sizexC

            # classification layer

            x = self.classify(gmerge) # batch_sizexnum_classes



        else:
            c1, c2, c3 = None, None, None
            x = self.classify(torch.squeeze(g))

        return [x, c1, c2, c3]



