# src/models/gp_flow.py
import gpytorch.constraints
import torch
import gpytorch.likelihoods
import gpytorch.means
import gpytorch.kernels
import gpytorch.distributions

class GP(gpytorch.models.ExactGP):
    def __init__(self):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise_covar.initialize(noise=0.1)
        super().__init__(None, None, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()  
        
        ls_init = torch.tensor([2,0.5,0.5])
        lower = (ls_init * 0.25).clamp(min=1e-4)
        upper = (ls_init * 4)

        lengthscale_constraint = gpytorch.constraints.Interval(lower_bound=lower,
                                                               upper_bound=upper)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.keops.MaternKernel(
            nu=2.5,
            ard_num_dims=3,
            lengthscale_constraint=lengthscale_constraint
        ))
        #self.covar_module.base_kernel.raw_lengthscale.requires_grad_(False)


    def forward(self, TA):
        mean_x  = self.mean_module(TA)
        covar_x = self.covar_module(TA)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)