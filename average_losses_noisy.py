#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:21:34 2019

@author: debjani
"""
import torch
import matplotlib.pyplot as plt
from network_noisy import MAMLModel
from loss_on_random_task_noisy import loss_on_random_task

def average_losses(initial_model, PATH, n_samples, flag = True,  K=10, n_steps=10, optim = torch.optim.SGD):
    """
    returns the average learning trajectory of the model trained for ``n_iterations`` over ``n_samples`` tasks
    """

    #x = np.linspace(-5, 5, 2) # dummy input for test_on_new_task
    avg_losses = [0] * K
    for i in range(n_samples):
        if flag == True: 
            losses = loss_on_random_task.loss(initial_model, PATH, K, n_steps,optim)
            avg_losses = [l + l_new for l, l_new in zip(avg_losses, losses)]
        else:
            losses = loss_on_random_task.loss_pretrain(initial_model, PATH, K, n_steps,optim)
            avg_losses = [l + l_new for l, l_new in zip(avg_losses, losses)]
            
    avg_losses = [l / n_samples for l in avg_losses]
    
    return avg_losses

model = MAMLModel()
PATH_MAML = "models/model.pth"
PATH_BASE = "models/base_model.pth"
plt.plot(average_losses(model, PATH_MAML, n_samples=50, flag = True, K=10, n_steps=10, optim=torch.optim.Adam), label='maml')
plt.plot(average_losses(model, PATH_BASE, n_samples=50, flag = False, K=10, n_steps=10, optim=torch.optim.Adam), label='pretrained')
plt.legend()
plt.title("Average learning trajectory for K=10, starting from initial weights")
plt.xlabel("gradient steps taken with Adam")
plt.show()