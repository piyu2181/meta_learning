#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 18:46:52 2019

@author: debjani
"""
############Import required functions##########################################
import torch.nn as nn
from collections import OrderedDict
############Import required functions##########################################

class MAMLModel(nn.Module):
    def __init__(self):
        super(MAMLModel, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(1,40)),
            ('relu1', nn.ReLU())
        ]))
        self.modela = nn.Sequential(OrderedDict([
            ('l2a', nn.Linear(40,40)),
            ('relu2a', nn.ReLU()),
            ('l3a', nn.Linear(40,1))
        ]))
        self.modelb = nn.Sequential(OrderedDict([
            ('l2b', nn.Linear(40,40)),
            ('relu2b', nn.ReLU()),
            ('l3b', nn.Linear(40,1))
        ]))
        
    def forward(self, x):
        return self.modela(self.model(x)), self.modelb(self.model(x))
    
    def parameterised(self, x, weights):
        # like forward, but uses ``weights`` instead of ``model.parameters()``
        # it'd be nice if this could be generated automatically for any nn.Module...
        x = nn.functional.linear(x, weights[0], weights[1])
        x = nn.functional.relu(x)
        xa = nn.functional.linear(x, weights[2], weights[3])
        xa = nn.functional.relu(xa)
        xa = nn.functional.linear(xa, weights[4], weights[5])
        xb = nn.functional.linear(x, weights[6], weights[7])
        xb = nn.functional.relu(xb)
        xb = nn.functional.linear(xb, weights[8], weights[9])
        return xa, xb
    

                        