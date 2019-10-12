#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 19:36:48 2019

@author: debjani
"""
############Import required functions##########################################
import torch
import numpy as np
from network_noisy import MAMLModel
from meta_training_noisy import MAML
from src.sine_tasks_uncertainty import Sine_Task_Distribution
###############################################################################
def main():
    
    #sample tasks
    tasks = Sine_Task_Distribution(0.1, 5, 0, np.pi, -5, 5, 0, np.pi)
    maml = MAML(MAMLModel(), tasks, inner_lr=0.001, meta_lr=0.001)
    maml.main_loop(num_iterations=10000)
    # save the model
    torch.save(maml.model.state_dict(), 'models/model_30_10000.pth')
    
if __name__ == '__main__':
    main()