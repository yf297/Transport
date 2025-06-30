# src/models/gp_flow.py
import gpytorch.constraints
import torch
import gpytorch.likelihoods
import gpytorch.means
import gpytorch.kernels
import gpytorch.distributions
import torch.nn as nn
import torch.nn.functional as F

class TransportKernel(gpytorch.kernels.Kernel):
    def __init__(self, flow):
        super().__init__()
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.MaternKernel(
            nu = 1.5,
            ard_num_dims = 3,
            active_dims =(0,1,2)
        )) 
        self.kernel.base_kernel.lengthscale = torch.tensor([1.1, 0.5, 0.5])
        self.flow = flow

    def forward(self, TXY1, TXY2, **kwargs):
        T1 = TXY1[..., 0:1] 
        T2 = TXY2[..., 0:1]
        
        A1 = self.flow(TXY1)
        A2 = self.flow(TXY2)
        
        TA1 = torch.cat([T1, A1], dim=-1)
        TA2 = torch.cat([T2, A2], dim=-1)

        return self.kernel(TA1, TA2) 


class TransportGP(gpytorch.models.ExactGP):
    def __init__(self, flow):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(0.0))
        likelihood.noise = torch.tensor(0.1)
        super().__init__(None, None, likelihood)
        self.flow = flow
        self.mean = gpytorch.means.ConstantMean()
        self.kernel = TransportKernel(self.flow)

    def forward(self, TXY):
        mean = self.mean(TXY)
        kernel = self.kernel(TXY)
        return gpytorch.distributions.MultivariateNormal(mean, kernel)
    
