#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 20:39:25 2019

@author: debjani
"""
############Import required functions##########################################
import torch
import numpy as np
############I##################################################################
class Sine_Task():
    """
    A sine wave data distribution object with interfaces designed for MAML.
    """
    
    def __init__(self, amplitude, phase, amplitude_noise, phase_noise, xmin, xmax):
        self.amplitude = amplitude
        self.amplitude_noise= amplitude_noise
        self.phase = phase
        self.phase_noise = phase_noise
        self.xmin = xmin
        self.xmax = xmax
        
    def true_function(self, x):
        """
        Compute the true function and noise function on the given x.
        """
        sine_freq = np.random.uniform(1,2,1)
        noise_freq = np.random.uniform(0.1,0.5,1)
        y_true = self.amplitude * np.sin(self.phase + x * sine_freq)
        # adding noise 
        y2 = self.amplitude_noise * np.sin(self.phase_noise + (x)*noise_freq)# sigma
        #kernel = (y2 * y2)*np.identity(x.shape[0])
        ydist = np.random.normal(0,1, x.shape[0])
        #y_noise = np.matmul(y2, kernel)
        noise = (y2*y2)*0.5 * ydist
        #y_noisy = y + y_noise
        
        return (y_true, noise, y2 )
        
    def sample_data(self, target_flag = True, size=1):
        """
        Sample data from this task.
        
        returns: 
            x: the feature vector of length size
            y: the target vector of length size
            y2: the noise vector of length size
        """
        self.target_flag = target_flag
        x = np.random.uniform(self.xmin, self.xmax, size)
        (y_true, noise, sigma) = self.true_function(x)
        y_noisy = y_true + noise 
        
        x = torch.tensor(x, dtype=torch.float).unsqueeze(1)
        #print("x_shape:", x.shape)
        y_true = torch.tensor(y_true, dtype=torch.float).unsqueeze(1)
        #print("y_shape", y.shape)
        y_noisy =  torch.tensor(y_noisy, dtype=torch.float).unsqueeze(1)
        #print("y_shape", y2.shape)
        #sigma-band
        sigma =  torch.tensor(sigma, dtype=torch.float).unsqueeze(1)
        noise =  torch.tensor(noise, dtype=torch.float).unsqueeze(1)
        
        if target_flag:
            context_x = x[:size]
            context_y_true = y_true[:size]
            context_y_noisy = y_noisy[:size]
            context_sigma = sigma[:size]
            context_noise = noise[:size]
            return (context_x, context_y_true, context_y_noisy, context_sigma, context_noise )
        
#        else:
#          
#            context_x = x[-size:]
#            context_y = y[-size:]
#            context_y2 = y2[-size]
#            return (context_x, context_y, context_y2)
        
        #return(x, y, y2)
        
        
    
class Sine_Task_Distribution():
    """
    The task distribution for sine regression tasks for MAML
    """
    
    def __init__(self, amplitude_min, amplitude_max, phase_min, phase_max, x_min, x_max,
                 amplitude_min_noise, amplitude_max_noise, phase_min_noise, phase_max_noise):
        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max
        self.phase_min = phase_min
        self.phase_max = phase_max
        self.x_min = x_min
        self.x_max = x_max
        self.amplitude_min_noise = amplitude_min_noise
        self.amplitude_max_noise = amplitude_max_noise
        self.phase_min_noise = phase_min_noise
        self.phase_max_noise = phase_max_noise
        #self.number_of_points = number_of_points
        
    def sample_task(self):
        """
        Sample from the task distribution.
        
        returns:
            Sine_Task object
        """
        amplitude = np.random.uniform(self.amplitude_min, self.amplitude_max)
        phase = np.random.uniform(self.phase_min, self.phase_max)
        
        ''' Adding noise to y '''
        a = np.random.uniform(0,1,1) # a is the scale for amplitude
        amplitude_noise = amplitude * a
        phase_noise = np.random.uniform(self.phase_min_noise, self.phase_max_noise)
        #x = np.random.uniform(self.x_min, self.x_max, self.number_of_points)
        
       #obj1 =  Sine_Task(amplitude, phase, self.x_min, self.x_max, x)
       # obj2 = Sine_Task(amplitude_noise, phase_noise, self.x_min, self.x_max, x)
        
        
        return Sine_Task(amplitude, phase, amplitude_noise, phase_noise, self.x_min, self.x_max )