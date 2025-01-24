import torch
import gpytorch
import net

class GP(gpytorch.models.ExactGP):
    def __init__(self, xyt, z, flow, likelihood):
        super(GP, self).__init__(xyt, z, likelihood)
        self.flow = flow
        self.Mean = net.Mean(L = 3)
        self.Kernel = gpytorch.kernels.ScaleKernel(
                      gpytorch.kernels.MaternKernel(
                      nu = 1/2, ard_num_dims = 2))
        
    def forward(self, xyt):
        a = self.flow(xyt)
        mean = self.Mean(a)
        covar = self.Kernel(a, a)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class GP_FLOW:
    def __init__(self, xyt, z, flow, vel):
        self.flow = flow
        self.vel = vel
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.gp = GP(xyt, z, flow, self.likelihood)