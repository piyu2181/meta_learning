"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

############Import required functions##########################################
import torch
import torch.nn as nn
############Import required functions##########################################
class MAML():
    def __init__(self, model, tasks, inner_lr, meta_lr, K=30, inner_steps=1, tasks_per_meta_batch=1000):
        
        # important objects
        self.tasks = tasks
        self.model = model
        # Puting model on gpu if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.weights = list(model.parameters())
        self.criterion = nn.MSELoss()
        self.meta_optimiser = torch.optim.Adam(self.weights, meta_lr)
    
        
        # hyperparameters
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.K = K
        self.inner_steps = inner_steps 
        self.tasks_per_meta_batch = tasks_per_meta_batch 
        
        # metrics
        self.plot_every = 5
        self.print_every = 5
        self.meta_losses = []
        self.meta_mean_losses = []
        self.meta_sigma_losses = []
    
    def inner_loop(self, task):
        # reset inner model to current maml weights
        temp_weights = [w.clone() for w in self.weights]
        
        # perform training on data sampled from task
        X, y = task.sample_data(True,self.K)
        X = X.to(self.device)
        y = y.to(self.device)
    
        for step in range(self.inner_steps):
            mean_loss = self.criterion(self.model.parameterised(X, temp_weights)[0], y) / self.K # kind of training loss
            sigma_loss = self.criterion(self.model.parameterised(X, temp_weights)[1] , torch.abs(y - self.model.parameterised(X, temp_weights)[0])) / self.K # kind of training loss
            final_loss = (mean_loss + sigma_loss)
            
            # compute grad and update inner loop weights
            grad =torch.autograd.grad(final_loss, temp_weights)
            #grad_noise =torch.autograd.grad(noise_loss, temp_noise_weights)
            temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]
            
        
        #sample new data for meta-update and compute loss
        X, y =  task.sample_data(True, self.K)
        X = X.to(self.device)
        y = y.to(self.device)
        mean_loss = self.criterion(self.model.parameterised(X, temp_weights)[0], y) / self.K # kind of training loss
        sigma_loss = self.criterion(self.model.parameterised(X, temp_weights)[1] , torch.abs(y - self.model.parameterised(X, temp_weights)[0])) / self.K
        final_loss = mean_loss + sigma_loss
        return (final_loss, mean_loss.item(), sigma_loss.item())
    
    def main_loop(self, num_iterations):
        epoch_loss = 0
        epoch_mean_loss = 0
        epoch_sigma_loss = 0
        
        for iteration in range(1, num_iterations+1):
            
            # compute meta loss
            meta_loss = 0
            meta_mean_loss = 0
            meta_sigma_loss = 0
            
            for i in range(self.tasks_per_meta_batch):
                task = self.tasks.sample_task()
                a,b,c = self.inner_loop(task)
                meta_loss += a
                meta_mean_loss += b
                meta_sigma_loss += c
            
            # compute meta gradient of loss with respect to maml weights
            meta_grads = torch.autograd.grad(meta_loss, self.weights)
            
            # assign meta gradient to weights and take optimisation step
            for w, g in zip(self.weights, meta_grads):
                w.grad = g
            self.meta_optimiser.step()
            
            
            # log metrics
            epoch_loss += meta_loss.item() / self.tasks_per_meta_batch
            epoch_mean_loss += meta_mean_loss / self.tasks_per_meta_batch
            epoch_sigma_loss += meta_sigma_loss / self.tasks_per_meta_batch
            
            if iteration % self.print_every == 0:
                print("{}/{}. loss: {}".format(iteration, num_iterations, epoch_loss / self.plot_every))
                print("{}/{}. y_loss: {}".format(iteration, num_iterations, epoch_mean_loss / self.plot_every))
                print("{}/{}. noise_loss: {}".format(iteration, num_iterations, epoch_sigma_loss / self.plot_every))
                
            
            if iteration % self.plot_every == 0:
                self.meta_losses.append(epoch_loss / self.plot_every)
                self.meta_mean_losses.append(epoch_mean_loss / self.plot_every)
                self.meta_sigma_losses.append(epoch_sigma_loss / self.plot_every)
                epoch_loss = 0
                epoch_mean_loss = 0
                epoch_sigma_loss = 0