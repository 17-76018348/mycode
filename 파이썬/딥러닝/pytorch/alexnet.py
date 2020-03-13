# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:52:42 2020

@author: didrm
"""
import torch
import torch.nn as nn
import torch.optim as optim







class Alexnet(nn.Module):
    def __init__(self,output_classes):
        super(Alexnet,self).__init()
        self.con = nn.Sequential(
                    nn.Conv2d(in_channels = 3, out_channels = 96,
                              kernel_size = (11, 11), stride = 4),
                    nn.ReLU(True),
                    nn.MaxPool2d(kernel_size = (3,3),stride = 2),
                    
                    nn.Conv2d(in_channels = 96, out_channels = 256,
                              kernel_size = (5,5), stride = 1, padding = 2),
                    nn.ReLU(True),
                    nn.MaxPool2d(kernel_size = (3, 3), stride = 2),
                    
                    nn.Conv2d(in_channels = 256, out_channels = 384,
                              kernel_size = (3,3), padding = 1),
                    nn.ReLU(True),
                    
                    nn.Conv2d(in_channels = 384, out_channels = 384,
                              kernel_size = (3,3), padding = 1),
                    nn.ReLU(True),
                    
                    nn.Conv2d(in_channels = 384, out_channels = 256,
                              kernel_size = (3,3), padding = 1),
                    nn.ReLU(True),
                    nn.MaxPool2d(kernel_size = (3,3), stride = 2)
            
            )
        
        
        self.fcl = nn.Sequential(
                    nn.Linear(256*6*6,4096),
                    nn.ReLU(True),
                    nn.Linear(4096,4096),
                    nn.ReLU(True),
                    nn.Linear(4096,output_classes),
            )
        