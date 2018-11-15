#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:58:21 2018

@author: ws
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
from torch.nn import Sequential


class CNN( nn.Module ):
    def __init__( self, num_classes ):
        super( CNN, self ).__init__()
        
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16.classifier = nn.Sequential(nn.Linear(512*8*8,4096),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(4096,4096),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(4096,num_classes))
#        self.init_weights()
        
    def init_weights( self ):
        """Initialize the weights."""
        init.kaiming_uniform( self.linear.weight, mode='fan_in' )
        self.linear.bias.data.fill_( 0 )
        
        
    def forward( self, images ):

        
        # Last conv layer feature map
        x = self.vgg16( images )
#        x = Variable(x.data)
#        x = self.classifier(x)
        return x