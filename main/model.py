import torch
import gpytorch
import matplotlib.pyplot as plt
from . import plot, ode, tools, indices
import math

class SpaceTimeKernel(gpytorch.kernels.Kernel):
    def __init__(self, l0, l1, l2,  **kwargs):
        super(SpaceTimeKernel, self).__init__(**kwargs)

        self.kernel = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.keops.MaternKernel(nu=5/2,
                                          ard_num_dims=3))

        self.kernel.base_kernel.initialize(lengthscale=[l0, l1, l2])

    def forward(self, ta0, ta1, **params):
        return self.kernel(ta0, ta1) 
    
class GPFlow(gpytorch.models.ExactGP):
    def __init__(self, spaceTimeKernel, flow):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super(GPFlow, self).__init__(None, None, likelihood)
        self.mean = gpytorch.means.ZeroMean()
        self.kernel = spaceTimeKernel
        self.flow = flow
        
    def forward(self, TXY):
        unique_T = torch.unique(TXY[:, 0])
        list_of_XY = [TXY[TXY[:, 0] == t][:, 1:] for t in unique_T]
        TA = [torch.cat([
                unique_T[i].repeat(list_of_XY[i].shape[0], 1),
                self.flow(unique_T[i], list_of_XY[i])
            ], dim=-1) for i in range(unique_T.shape[0])]
        
        TA = torch.cat(TA, dim = 0)

        mean = self.mean(TA)
        covar = self.kernel(TA)
        return gpytorch.distributions.MultivariateNormal(mean, covar)



class data():
    def __init__(self, 
                 T, XY, Z, 
                 XY_UV = None,
                 grid_size = 1, k0 = 0, k1 = 1):
       
        self.T = T
        self.XY = XY
        self.Z = Z
        self.XY_UV = XY_UV
        
        self.n = self.T.shape[0]
        self.m = self.XY.shape[0]
        
        self.normalizeTime = tools.NormalizeTime(T)
        self.normalizeSpace = tools.NormalizeSpace(XY)
        self.normalizeObs = tools.NormalizeObs(Z)
        
        self.grid_size = grid_size
        self.prediction = indices.Prediction(self.normalizeTime(self.T),
                        self.normalizeSpace(self.XY), 
                        self.normalizeObs(self.Z), 
                        grid_size = self.grid_size)
        
        self.conditining = indices.Conditioning(self.normalizeTime(self.T),
                         self.normalizeSpace(self.XY), 
                         self.normalizeObs(self.Z), 
                         grid_size = self.grid_size, k0 = k0, k1 = k1)
        
        
        self.cells = self.conditining.cells
        
                
        self.extent = None
        self.date = None
        self.level = None
        self.id = None
        self.minutes = None
        
        
        
    def plot_observations(self, 
                          frame = 0, 
                          gif = False):

        XY = self.XY
        Z = self.Z
          
        return plot.observations(XY, Z, self.extent, frame, gif)
    
    def plot_vel_data(self,
                      indices = None,
                      scale = 1,
                      frame = 0,
                      color = "blue",
                      gif = False):
        
        XY_UV = self.XY_UV
        
        if indices is not None:
            XY_UV_subset =  [XY_UV[i][indices,:] for i in range(0,self.n)]
            XY_UV = XY_UV_subset

        return plot.velocities(XY_UV, self.extent, scale, frame, color, gif)
    
    def plot_vel(self, 
                 indices = None,
                 scale = 1,
                 frame = 0, 
                 color = "blue",
                 gif = False):
        
        T = self.T
        XY = self.XY
        
        if indices is not None:
            XY = XY[indices,:]
            
        UV = torch.ones((self.n, XY.shape[0], 2))
        
        vel = ode.Vel(self.flow, self.normalizeTime, self.normalizeSpace)
        for frame in range(0,self.n):
            UV[frame,:,:] = vel(T[frame], XY)
        
        XY_UV = [torch.cat([XY,
                            UV[i,:,0:1],
                            UV[i,:,1:2]],
                dim = -1).detach() for i in range(0,self.n)]
        
        return plot.velocities(XY_UV, self.extent, scale, frame, color, gif)