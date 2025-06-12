# src/models/gp_flow.py
import gpytorch.constraints
import torch
import gpytorch.likelihoods
import gpytorch.means
import gpytorch.kernels
import gpytorch.distributions

class TransportKernel(gpytorch.kernels.Kernel):
    def __init__(self, flow):
        super().__init__()
        self.kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.keops.MaternKernel(
            nu=2.5,
            ard_num_dims=3
        ))
        self.kernel.base_kernel.lengthscale = torch.tensor([2.0, 0.3, 0.3]) 
        self.kernel.outputscale = torch.tensor(1.0)
        self.flow = flow
    

    def forward(self, TXY1, TXY2, **kwargs):
        TA1 = self.flow(TXY1)
        TA2 = self.flow(TXY2)
        return self.kernel(TA1, TA2) 
    
    
class TransportGP(gpytorch.models.ExactGP):
    def __init__(self, flow):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = torch.tensor(0.1)
        super().__init__(None, None, likelihood)

        self.flow = flow
        self.mean = gpytorch.means.ConstantMean()
        self.kernel = TransportKernel(self.flow)

    def forward(self, TXY):
        mean = self.mean(TXY)
        kernel = self.kernel(TXY)
        return gpytorch.distributions.MultivariateNormal(mean, kernel)