import torch
import gpytorch

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
    def __init__(self, T, XY, Z, XY_UV = None):
        self.T = T
        self.XY = XY
        self.Z = Z
        self.XY_UV = XY_UV
        
        self.extent = None
        self.date = None
        self.level = None
        
        self.indices = None
        self.gp = None
        self.flow = None
        
        self.input_std = torch.ones(1)
        self.output_std = torch.ones(1)
        
        self.n = self.T.shape[0]
        self.m = self.XY.shape[0]
        
    def plot_observations(self, indices, frame = 0):
        XY_ = self.XY
        Z_ = self.Z
     
        if indices is not None:
            XY_ = XY_[indices,:]
            Z_ =  [Z[indices] for Z in Z_]
            
        return plot.observations(XY_, Z_, self.extent, frame)
    
    
    
    
    
    
    def plot_vel_data(self, indices, frame = 0):
        T_ = self.T
        XY_UV_ = self.XY_UV
        XY_UV =  [XY_UV_[i][indices,:] for i in range(0,T_.shape[0])]
        return plot.velocities(XY_UV, self.extent, frame)
    
    
    def plot_vel(self, indices, frame = 0):
        T_ = self.T
        XY_ = self.XY
        
        XY_ = XY_[indices, ]
        UV = torch.ones((T_.shape[0], XY_.shape[0], 2))
        
        vel = ode.Vel_hat(self)
        for frame in range(0,T_.shape[0]):
            UV[frame,:,:] = vel(T_[frame], XY_)
        
        XY_UV = [torch.cat([XY_,
                            UV[i,:,0:1],
                            UV[i,:,1:2]],
                dim = -1).detach() for i in range(0,T_.shape[0])]
        return plot.velocities(XY_UV, self.extent, frame)
       