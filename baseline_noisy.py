#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:17:39 2019

@author: debjani
"""
import torch.nn as nn
import torch
import numpy as np
from network_noisy import MAMLModel
from src.sine_tasks import Sine_Task_Distribution

tasks = Sine_Task_Distribution(0.1, 5, 0, np.pi, 0.1, 2, 0, (np.pi), -5, 5,)
class baseline():
    def mixed_pretrained(iterations=500):
        """
        returns a model pretrained on a selection of ``iterations`` random tasks.
        """
        # set up model
        model = MAMLModel()
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        # fit the model
        for i in range(iterations):
            model.zero_grad()
            x, y, y2 = tasks.sample_task().sample_data(True, 10)
            y_noise = (y2-y)
            ypred, y_noise_pred = model(x)
            loss = criterion(ypred, y)
            noise_loss = criterion(y_noise_pred, y_noise)
            total_loss = loss + noise_loss
            total_loss.backward()
            optimiser.step()       
        return model

pretrained = baseline.mixed_pretrained(1000)
# save the model
torch.save(pretrained.state_dict(), 'models/base_model.pth')