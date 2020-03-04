#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 21:01:16 2020

@author: yki
"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class cornea(Dataset):
    def __init__(self, X, y=None, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        
    def __len__(self):
        return len(self.X.index)
    
    def __getitem__(self, index):
#         image = self.X.iloc[index, ].values.astype(np.uint8).reshape((28, 28, 1))
        image = self.X.iloc[index, ].values.astype(np.float32).reshape((29,1))
        
        if self.transform is not None:
            image = self.transform(image)
            
        if self.y is not None:
            return image, self.y.iloc[index]
        else:
            return image
        
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(29, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256,4)
        )
        
    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = x.view(x.size(0), -1)
        # print('shape, type of x in MLP__forward__:',x.shape,type(x))
        x = self.layers(x)
        x1 = F.log_softmax(x,dim=-1)
        x2 = F.softmax(x,dim=-1)
        return x1,x2