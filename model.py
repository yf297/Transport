
import torch
import gpytorch
import net

class Mean(gpytorch.means.Mean):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean_W = net.Mean(L=4)
        net.initialize(self.mean_W)

    def forward(self, x, **params):
        # Requires PyTorch >= 2.0 for torch.vmap
        m = torch.vmap(self.mean_W)(x).squeeze()
        return m


class GP(gpytorch.models.ExactGP):
    def __init__(self, x, y, likelihood):
        super().__init__(x, y, likelihood)
        self.mean_module = Mean()
        self.kernel_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=3/2, ard_num_dims=2)
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.kernel_module(x, x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class Initialize:
    def __init__(self, Flow, Vel):
        self.Flow = Flow
        self.Vel = Vel
        
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
        
        x = None
        y = None
        
        self.GP = GP(x, y, likelihood)