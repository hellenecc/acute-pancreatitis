import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from blocks import ConvBlock, LinearAttentionBlock,GridAttentionBlock, ProjectorBlock
from initialize import *

'''
attention before max-pooling
'''
#vgg16
class Clinicmodel(nn.Module):
    def __init__(self, data_size, num_classes, init='xavierUniform'):
        super(Clinicmodel, self).__init__()
        self.linear1=nn.Linear(in_=32, out_=16, bias=True)
        self.linear2 = nn.Linear(in_=16, out_=8, bias=True)
        self.classify = nn.Linear(in_=16, out_=2, bias=True)
        # Projectors & Compatibility functions

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
        x = self.linear1(x)
        #x = self.linear2(x)
        x = self.classify(x) # batch_sizexnum_classes

        return x



