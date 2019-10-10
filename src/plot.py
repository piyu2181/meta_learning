"""
    To plot the sine functions and noise
"""


############Import required functions##########################################
import torch
import matplotlib.pyplot as plt
from sine_tasks import Sine_Task_Distribution
import numpy as np
############Import required functions##########################################

tasks = Sine_Task_Distribution(0.1, 5, 0, np.pi, -5, 5, 0.1, 2, 0, np.pi)
task = tasks.sample_task()

x, y, y2 = task.sample_data(True, 50)

x = torch.tensor(x, dtype=torch.float).squeeze(1)
y = torch.tensor(y, dtype=torch.float).squeeze(1)
y2 =  torch.tensor(y2, dtype=torch.float).squeeze(1)
   
plt_vals = []
xin = np.argsort(x)
plt_vals.extend([x[xin].numpy(), y[xin].numpy(), "k"])
plt_vals.extend([x[xin].numpy(), y2[xin].numpy(), "g"])

plt.plot(*plt_vals)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['y','noisy_y'])
plt.show()