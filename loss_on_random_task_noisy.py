
"""
Created on Tue Oct  8 16:30:01 2019

@author: debjani
"""

############Import required functions##########################################
import torch
import torch.nn as nn
import numpy as np
from src.sine_tasks import Sine_Task_Distribution

###############################################################################


class loss_on_random_task():
    def loss(initial_model, PATH,  K, num_steps, optim=torch.optim.SGD):
        """
        trains the model on a random sine task and measures the loss curve.
        
        for each n in num_steps_measured, records the model function after n gradient updates.
        """
        
        # copy MAML model into a new object to preserve MAML weights during training
        tasks = Sine_Task_Distribution(0.1, 5, 0, np.pi, 0.1, 2, 0, (np.pi), -5, 5,)
        # making an object of the network
        model = initial_model
        #print(model)
        maml_model = torch.load(PATH)
        #print("model_type:", model.type)
        #print(maml_model.keys())
        #print(initial_model.model.l1.bias)
        #print(maml_model)
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
        optimiser = optim(model.parameters(), 0.01)
    
        # train model on a random task
        task = tasks.sample_task()
        X, y, y2 = task.sample_data(K)
        y_noise = (y2 - y)
        losses = []
        noise_losses = []
        total_losses = []
        for step in range(1, num_steps+1):
            y_pred, y_noise_pred = model(X)
            loss = criterion(y_pred, y) / K
            noise_loss = criterion(y_noise_pred, y_noise) / K
            total_loss = (loss + noise_loss)
            total_losses.append(total_loss.item())
            losses.append(loss.item())
            noise_losses.append(noise_loss.item())
    
            # compute grad and update inner loop weights
            model.zero_grad()
            total_loss.backward()
            optimiser.step()
            
        return total_losses

    def loss_pretrain(initial_model, PATH,  K, num_steps, optim=torch.optim.SGD):
        """
        trains the model on a random sine task and measures the loss curve.
        
        for each n in num_steps_measured, records the model function after n gradient updates.
        """
        
        # copy MAML model into a new object to preserve MAML weights during training
        tasks = Sine_Task_Distribution(0.1, 5, 0, np.pi, 0.1, 2, 0, (np.pi), -5, 5,)
        # making an object of the network
        model = initial_model
        model.load_state_dict(torch.load(PATH))
        criterion = nn.MSELoss()
        optimiser = optim(model.parameters(), 0.01)
    
        # train model on a random task
        task = tasks.sample_task()
        X, y, y2 = task.sample_data(K)
        y_noise = (y2 - y)
        losses = []
        noise_losses = []
        total_losses = []
        for step in range(1, num_steps+1):
            y_pred, y_noise_pred = model(X)
            loss = criterion(y_pred, y) / K
            noise_loss = criterion(y_noise_pred, y_noise) / K
            total_loss = (loss + noise_loss)
            total_losses.append(total_loss.item())
            losses.append(loss.item())
            noise_losses.append(noise_loss.item())
    
            # compute grad and update inner loop weights
            model.zero_grad()
            total_loss.backward()
            optimiser.step()
            
        return total_losses