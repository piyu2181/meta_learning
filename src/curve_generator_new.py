'''
  author : Debjani
  This code will generate random functions from gaussian process and will add
  non iid noise to each function.
  Inputs: tasks_per_meta_batch : number of random functions to be generated.
          num_total_points: maximum number of points in each 
          function(size of x data)
  output : y = f(x) + epsilon
'''


import torch
import random


class Gaussian_Curve():
    """
    Gaussian curve object with interfaces designed for MAML.
    """
    
    def __init__(self, tasks_per_meta_batch, num_total_points,
                     x_size=1, l1_scale=0.4,
                     sigma_scale=1.0 ):
           
        
        """Creates a gaussian dataset of functions sampled from a GP.
        
        Args:
          tasks_per_meta_batch : An integer.
          num_total_points : The max number of observations in the context.
          x_size: Integer >= 1 for length of "x values" vector.
          y_si  ze: Integer >= 1 for length of "y values" vector.
          l1_scale: Float; typical scale for kernel distance function.
          sigma_scale: Float; typical scale for variance.
        """
        
        
        self.tasks_per_meta_batch = tasks_per_meta_batch
        self.num_total_points = num_total_points
        self.x_size = x_size
        self.l1_scale = l1_scale
        self.sigma_scale = sigma_scale
        #self.target_flag = target_flag

        
    def gaussian_kernel(self, xdata, l1, sigma_f, sigma_noise=2e-2):
        
        """Applies the Gaussian kernel to generate curve data.
        
        Args:
          xdata: Tensor with shape `[tasks_per_meta_batch, num_total_points, x_size]` with
          the values of the x-axis data.
          l1: Tensor with shape `[tasks_per_meta_batch, y_size, x_size]`, the scale
          parameter of the Gaussian kernel.
          sigma_f: Float tensor with shape `[tasks_per_meta_batch, y_size]`; the magnitude
          of the std.
          sigma_noise: Float, std of the noise that we add for stability.
        
        Returns:
          The kernel, a float tensor with shape
          `[tasks_per_meta_batch, y_size, num_total_points, num_total_points]`.
        """
        # Expand and take the difference
        xdata1 = xdata.unsqueeze(1)  # [B, 1, num_total_points, x_size]
        xdata2 = xdata.unsqueeze(2)  # [B, num_total_points, 1, x_size]
        ##print(xdata1.shape, xdata2.shape)
        diff = xdata1 - xdata2  # [B, num_total_points, num_total_points, x_size]

        # [B, y_size, num_total_points, num_total_points, x_size]
        norm = torch.mul(diff[:, None, :, :, :] / l1[:, :, None, None, :], diff[:, None, :, :, :] / l1[:, :, None, None, :])
        norm = torch.sum(
            norm, -1)  # [B, data_size, num_total_points, num_total_points]

        # [B, y_size, num_total_points, num_total_points]
        kernel = torch.mul(sigma_f, sigma_f)[:, :, None, None] * torch.exp(-0.5 * norm)

        ##print(kernel.shape)
        # Add some noise to the diagonal to make the cholesky work.
        kernel += (sigma_noise**2) * torch.eye(self.num_total_points)

        return kernel
    
    def true_function(self):
        
#       
        x_distribution = torch.distributions.uniform.Uniform(
              torch.tensor([-2.0]), torch.tensor([2.0]))
        xdata = x_distribution.sample(torch.Size(
              [self.tasks_per_meta_batch, self.num_total_points, self.x_size])).squeeze(-1)
        #print(xdata.shape)
        l1 = (torch.ones([self.tasks_per_meta_batch, self.x_size, self.x_size]) * 0.4)
        sigma_f = torch.ones([self.tasks_per_meta_batch, self.x_size]) * 1.0
        kernel = self.gaussian_kernel(xdata, l1, sigma_f)
    
        # Generating the y values
        xdata = xdata.unsqueeze(1)
    
        # Calculate Cholesky, using double precision for better stability:
        cholesky = torch.cholesky(kernel.double()).float()
    
        ydist = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        ydata = torch.matmul(
                cholesky,
                ydist.sample([self.tasks_per_meta_batch, self.x_size, self.num_total_points]))
        ##print('Ydata shape: ', ydata.shape)
    
        ###############################################################
        '''Adding noise to y'''
        temp1 = torch.rand(cholesky.shape[2:4])
        sigma = temp1 * torch.eye(self.num_total_points)
        ydist_noise = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        y_hat = torch.matmul(
            sigma,
            ydist_noise.sample([self.tasks_per_meta_batch, self.x_size, self.num_total_points]))
        ydata2 = torch.add(ydata, y_hat)

        #print('Y shape', ydata.shape)
        #print('Y data2 shape', ydata2.shape)
        
        '''NORMALIZING THE DATA'''
        #####################################################
        #YDATA
        max_val = torch.max(ydata)
        min_val = torch.min(ydata)
        ydata = torch.div(torch.sub(ydata, min_val),torch.sub(max_val,min_val))
    
        ydata2 = torch.div(torch.sub(ydata2, min_val),torch.sub(max_val,min_val))

        return(xdata, ydata, ydata2)
        
        
    def sample_data(self, target_flag = True, context_size = 1):
        """
        Sample data of size context_size from this task.
        target_flag = True : meta-tarining time
        target_flag = False: meta-validating time
        
        returns: 
        x: the feature vector of length size
        y: the target vector of length size
        y2: the noise vector of the length size
        
        """
        
        self.target_flag = target_flag
        self.xdata, self.y_data, self.y_data2 =   Gaussian_Curve.true_function(self)
        idx = torch.arange(1, self.num_total_points + 1)
        random.shuffle(idx)
        if target_flag:
            context_x = self.xdata[:, :context_size, :]
            context_y = self.y_data[:, :context_size, :]
            context_y2 = self.y_data2[:, :context_size, :]
            return (context_x, context_y, context_y2)
        
        else:
          
            context_x = self.xdata[:, -context_size:, :]
            context_y = self.y_data[:, -context_size:, :]
            context_y2 = self.y_data2[:, -context_size:, :]
            return (context_x, context_y, context_y2)

class Task_Distribution():
    """
    A gaussian curve data distribution object with interfaces designed for MAML.
    """
    def __init__(self, tasks_per_meta_batch, num_total_points=100, x_size=1, l1_scale=0.4, sigma_scale=1.0 ):

         
        self.tasks_per_meta_batch = tasks_per_meta_batch
        self.num_total_points = num_total_points
        self.x_size = x_size
        self.l1_scale = l1_scale
        self.sigma_scale = sigma_scale
        #self.target_flag = target_flag
     
    def sample_task(self):#, target_flag = True):
        """
        Compute the true function on the given x.
        """
        #self.target_flag = target_flag
        return(Gaussian_Curve(self.tasks_per_meta_batch, self.num_total_points, x_size= self.x_size, l1_scale= self.l1_scale, sigma_scale = self.sigma_scale)) #, self.target_flag))