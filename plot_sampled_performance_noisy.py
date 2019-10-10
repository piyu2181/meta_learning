#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:40:17 2019

@author: debjani
"""
from  model_functions_at_training_noisy import model_functions_at_training
import matplotlib.pyplot as plt
import torch
import numpy as np


class plot_sampled_performance():
    def plot_performance(initial_model, PATH,  model_name, X, y, y2, task, optim=torch.optim.SGD, lr=0.01, flag = True):
        x_axis = np.linspace(-5, 5, 1000)
        sampled_steps=[1,10]
        if flag== True:
            outputs_a, outputs_b, losses = model_functions_at_training.training(initial_model,PATH, 
                                                          X, y, y2,
                                                          sampled_steps=sampled_steps, 
                                                          x_axis=x_axis, 
                                                          optim=optim, lr=lr)
        else:
            outputs_a, outputs_b, losses = model_functions_at_training.training_baseline(initial_model,PATH, 
                                                          X, y, y2,
                                                          sampled_steps=sampled_steps, 
                                                          x_axis=x_axis, 
                                                          optim=optim, lr=lr)
            
    
        plt.figure(figsize=(15,5))
        
        # plot the model functions
        plt.subplot(1, 2, 1)
        plt.plot(x_axis, task.true_function(x_axis)[0], '-', color=(0, 0, 1, 0.5), label='true function')
        plt.scatter(X, y, label='data')
        y_noise = y2 - y
        y_noise_true = task.true_function(x_axis)[1] - task.true_function(x_axis)[0]
        plt.plot(x_axis, y_noise_true, '-', color=(0, 0, 1, 0.5), label='true function')
        plt.scatter(X, y_noise, label='noisy_data')
        plt.plot(x_axis, outputs_a['initial'], ':', color=(0.7, 0, 0, 1), label='initial weights_y')
        output_noise = outputs_b - outputs_a
        plt.plot(x_axis, output_noise['initial'], ':', color=(0.7, 0, 0, 1), label='initial weights_noise')
        
        for step in sampled_steps:
            plt.plot(x_axis, outputs_a[step], 
                     '-.' if step == 1 else '-', color=(0.5, 0, 0, 1),
                     label='model after {} steps'.format(step))
        for step in sampled_steps:
            plt.plot(x_axis, output_noise[step], 
                     '-.' if step == 1 else '-', color=(0.5, 0, 0, 1),
                     label='model after {} steps'.format(step))
            
                
        plt.legend(loc='lower right')
        plt.title("Model fit: {}".format(model_name))
    
        # plot losses
        plt.subplot(1, 2, 2)
        plt.plot(losses)
        plt.title("Loss over time")
        plt.xlabel("gradient steps taken")
        plt.show()