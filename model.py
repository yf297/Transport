import torch
import gpytorch
import net
from torchdiffeq import odeint

class GP(gpytorch.models.ExactGP):
    def __init__(self, TXY, Z, flow, likelihood):
        super(GP, self).__init__(TXY, Z, likelihood)
        self.flow = flow
        self.Mean = net.Mean(L = 4)
        self.Kernel = gpytorch.kernels.ScaleKernel(
                      gpytorch.kernels.MaternKernel(
                      nu = 1/2, ard_num_dims = 2))
        
    def forward(self, TXY):
        points = self.flow(TXY)
        mean = self.Mean(points)
        covar = self.Kernel(points, points)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class GP_FLOW:
    def __init__(self, TXY, Z, flow):
        
        self.TXY = TXY
        self.Z = Z
        
        self.flow = flow
        
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
        self.gp = GP(self.TXY, self.Z, self.flow, self.likelihood)