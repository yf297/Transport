import torch
import gpytorch
import matplotlib.pyplot as plt
from . import plot, ode

class GP(gpytorch.models.ExactGP):
    def __init__(self, kernel, likelihood):
        super(GP, self).__init__(None, None, likelihood)
        self.mean = gpytorch.means.ConstantMean()
        self.kernel = kernel

    def forward(self, points):
        mean = self.mean(points)
        covar = self.kernel(points, points)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class data():
    def __init__(self, T, XY, Z, XYP_UV = None, XY_full = None, Z_full = None):
        self.T = T
        self.XY = XY
        self.Z = Z
        self.XYP_UV = XYP_UV
        
        
        if XY_full is not None and Z_full is not None:
            self.XY_full = XY_full
            self.Z_full = Z_full
        else:
            self.XY_full = XY
            self.Z_full = Z
        
        self.output_mean = self.Z.reshape(-1).mean()
        self.output_normalize =  self.Z.reshape(-1).std()
        self.Z_normalized = [(z - self.output_mean)/self.output_normalize for z in self.Z]
        self.Z0_normalized =  self.Z_normalized[0]

        self.input_mean = self.XY.mean(dim=-2, keepdim=True)
        self.input_normalize = (self.XY - self.input_mean).max()
        self.XY_normalized = (self.XY - self.input_mean)/self.input_normalize
        
        self.time_normalize = self.T.max()
        self.T_normalized = self.T/self.time_normalize
        
        self.extent = None
        self.date = None
        self.level = None
        self.time = None
        self.id = None
        self.minutes = None
        
        self.flow = None
        self.vel = None
        self.n = self.T.shape[0]
        self.m = self.XY.shape[0]
        

        
    def plot_observations(self, 
                          frame = 0, 
                          full = True, 
                          gif = False):
        
        XY =  self.XY
        Z = self.Z
        if full:
            XY = self.XY_full
            Z = self.Z_full
          
        return plot.observations(XY, Z, self.extent, frame, gif)
    
    def plot_vel_data(self,
                      indices = None,
                      scale = 1,
                      frame = 0,
                      color = "blue",
                      gif = False):
        
        XY_UV_ = [torch.index_select(XYP_UV, dim=1, index=torch.tensor([0, 1, 3, 4]))
                  for XYP_UV in self.XYP_UV]
        
        if indices is not None:
            XY_UV =  [XY_UV_[i][indices,:] for i in range(0,self.n)]
        else:
            XY_UV = XY_UV_
            
        return plot.velocities(XY_UV, self.extent, scale, frame, color, gif)
    
    def plot_vel(self, 
                 indices = None,
                 scale = 1,
                 frame = 0, 
                 color = "blue",
                 gif = False):
        
        T_ = self.T
        XY_ = self.XY
        if indices is not None:
            XY_ = XY_[indices,:]
            
        UV = torch.ones((self.n, XY_.shape[0], 2))
        
        vel = ode.Vel_hat(self)

        for frame in range(0,self.n):
            UV[frame,:,:] = vel(T_[frame], XY_)
        
        XY_UV = [torch.cat([XY_,
                            UV[i,:,0:1],
                            UV[i,:,1:2]],
                dim = -1).detach() for i in range(0,self.n)]
        return plot.velocities(XY_UV, self.extent, scale, frame, color, gif)