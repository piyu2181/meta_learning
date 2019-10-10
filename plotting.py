#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 19:00:19 2019

@author: debjani
"""


from src.sine_tasks import Sine_Task_Distribution
from plot_sampled_performance_noisy import plot_sampled_performance
from network_noisy import MAMLModel
import numpy as np

tasks = Sine_Task_Distribution(0.1, 5, 0, np.pi,-5, 5, 0.1, 2, 0, (np.pi))
task = tasks.sample_task()
X,y,y2 = task.sample_data(True, 10)
K = 10

model = MAMLModel()
PATH_MAML = "models/model.pth"
PATH_BASE = "models/base_model.pth"

plot_sampled_performance.plot_performance(model, PATH_MAML,'MAML', X, y, y2, task, flag = True)
#plot_sampled_performance.plot_performance(model, PATH_BASE, 'Pretrained',  X, y, y2,task,flag = False)