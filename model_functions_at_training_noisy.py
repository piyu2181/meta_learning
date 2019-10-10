#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 18:25:42 2019

@author: debjani
"""
import torch
import torch.nn as nn

class model_functions_at_training(): 
    def training(initial_model, PATH, X, y, y2, sampled_steps, x_axis, optim=torch.optim.SGD, lr=0.01):
        """
        trains the model on X, y and measures the loss curve.
        
        for each n in sampled_steps, records model(x_axis) after n gradient updates.
        """
        
        # copy MAML model into a new object to preserve MAML weights during training
        model = initial_model
        maml_model = torch.load(PATH)
        initial_model.model.l1.weight.values = maml_model['model.l1.weight']
        initial_model.model.l1.bias.values = maml_model['model.l1.bias']
        initial_model.modela.l2a.weight.values = maml_model['model.l2a.weight']
        initial_model.modela.l2a.bias.values = maml_model['model.l2a.bias']
        initial_model.modela.l3a.weight.values = maml_model['model.l3a.weight']
        initial_model.modela.l3a.bias.values = maml_model['model.l3a.bias']
        initial_model.modelb.l2b.weight.values = maml_model['model.l2b.weight']
        initial_model.modelb.l2b.bias.values = maml_model['model.l2b.bias']
        initial_model.modelb.l3b.weight.values = maml_model['model.l3b.weight']
        initial_model.modelb.l3b.bias.values = maml_model['model.l3b.bias']
        criterion = nn.MSELoss()
        optimiser = optim(model.parameters(), lr)
    
        # train model on a random task
        num_steps = max(sampled_steps)
        K = X.shape[0]
        y_noise = (y2-y)
        losses = [] 
        noise_losses = []
        total_losses = []
        outputs_a = {}
        outputs_b = {}
        for step in range(1, num_steps+1):
            y_pred, y_noise_pred = model(X)
            loss = criterion(y_pred , y) / K
            noise_loss = criterion(y_noise_pred, y_noise) / K
            total_loss = (loss + noise_loss)
            total_losses.append(total_loss.item())
            losses.append(loss)
            noise_losses.append(noise_loss)
    
            # compute grad and update inner loop weights
            model.zero_grad()
            total_loss.backward()
            optimiser.step()
    
            # plot the model function
            if step in sampled_steps:
                #print( model(torch.tensor(x_axis, dtype=torch.float)[0:5]))
                outputs_a[step], outputs_b[step] = model(torch.tensor(x_axis, dtype=torch.float).view(-1, 1))
                outputs_a[step] =  outputs_a[step].detach().numpy()
                outputs_b[step] = outputs_b[step].detach().numpy()
                
        outputs_a['initial'], outputs_b['initial']= initial_model(torch.tensor(x_axis, dtype=torch.float).view(-1, 1))
        outputs_a['initial'] =  outputs_a['initial'].detach().numpy()
        outputs_a['initial'] =  outputs_b['initial'].detach().numpy()
        return outputs_a, outputs_b, total_losses
    
    def training_baseline(initial_model, PATH, X, y, y2, sampled_steps, x_axis, optim=torch.optim.SGD, lr=0.01):
        """
        trains the model on X, y and measures the loss curve.
        
        for each n in sampled_steps, records model(x_axis) after n gradient updates.
        """
        
        # copy MAML model into a new object to preserve MAML weights during training
        model = initial_model
        initial_model.load_state_dict((torch.load(PATH)))
        #model.load_state_dict(initial_model.state_dict())
        criterion = nn.MSELoss()
        optimiser = optim(model.parameters(), lr)
    
        # train model on a random task
        num_steps = max(sampled_steps)
        K = X.shape[0]
        y_noise = (y2-y)
        losses = [] 
        noise_losses = []
        total_losses = []
        outputs_a = {}
        outputs_b = {}
        for step in range(1, num_steps+1):
            y_pred, y_noise_pred = model(X)
            loss = criterion(y_pred , y) / K
            noise_loss = criterion(y_noise_pred, y_noise) / K
            total_loss = (loss + noise_loss)
            total_losses.append(total_loss.item())
            losses.append(loss)
            noise_losses.append(noise_loss)
    
            # compute grad and update inner loop weights
            model.zero_grad()
            total_loss.backward()
            optimiser.step()
    
            # plot the model function
            if step in sampled_steps:
                outputs_a[step], outputs_b[step] = model(torch.tensor(x_axis, dtype=torch.float).view(-1, 1))
                outputs_a[step] =  outputs_a[step].detach().numpy()
                outputs_b[step] = outputs_b[step].detach().numpy()
                
        outputs_a['initial'], outputs_b['initial']= initial_model(torch.tensor(x_axis, dtype=torch.float).view(-1, 1))
        outputs_a['initial'] =  outputs_a['initial'].detach().numpy()
        outputs_a['initial'] =  outputs_b['initial'].detach().numpy()
        return outputs_a, outputs_b, total_losses
    