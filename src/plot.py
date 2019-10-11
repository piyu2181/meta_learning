"""
    To plot the sine functions and noise
"""


############Import required functions##########################################
import torch
import matplotlib.pyplot as plt
from sine_tasks_uncertainty import Sine_Task_Distribution
import numpy as np
############Import required functions##########################################

tasks = Sine_Task_Distribution(0.1, 5, 0, np.pi, -5, 5, 0.1, 2, 0, np.pi)
task = tasks.sample_task()

x, y_true, y_noisy, sigma, noise = task.sample_data(True, 50)
#y_noisy = torch.sub(y_noisy,y_true)
sigma_pred = sigma + y_true
x = torch.tensor(x, dtype=torch.float).squeeze(1)
y_true = torch.tensor(y_true, dtype=torch.float).squeeze(1)
y_noisy =  torch.tensor(y_noisy, dtype=torch.float).squeeze(1)
sigma =  torch.tensor(sigma, dtype=torch.float).squeeze(1)
noise = torch.tensor(noise, dtype = torch.float).squeeze(1)
sigma_pred =  torch.tensor(sigma_pred, dtype=torch.float).squeeze(1)
   
plt_vals = []
xin = np.argsort(x)
plt_vals.extend([x[xin].numpy(), y_true[xin].numpy(), "k"])
plt_vals.extend([x[xin].numpy(), y_noisy[xin].numpy(), "r^"])
#plt_vals.extend([x[xin].numpy(), sigma[xin].numpy(), "g"])
#plt_vals.extend([x[xin].numpy(), sigma[xin].numpy()*(-1), "g"])
#plt_vals.extend([x[xin].numpy(), noise[xin].numpy(), "b"])
plt_vals.extend([x[xin].numpy(), sigma_pred[xin].numpy(), "g"])
plt_vals.extend([x[xin].numpy(), sigma_pred[xin].numpy()*(-1), "g"])

plt.plot(*plt_vals)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['y_true','noisy_y','variance-band','variance-band', 'noise'])
plt.show()