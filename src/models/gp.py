# src/models/gp_flow.py
import torch
import gpytorch.likelihoods
import gpytorch.means
import gpytorch.kernels
import gpytorch.distributions

class GP(gpytorch.models.ExactGP):
    def __init__(self, flow, l_0, l_1, l_2):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.initialize(noise=0.1)
        super().__init__(None, None, likelihood)
        self._flow = flow  
        self.mean_module = gpytorch.means.ConstantMean()  

        ls_init = torch.tensor([l_0, l_1, l_2])
        lower = (ls_init * (1/4)).clamp(min=1e-4)
        upper = (ls_init * 4).clamp(min=1e-4)

        matern = gpytorch.kernels.keops.MaternKernel(
            nu=2.5,
            ard_num_dims=3,
            lengthscale_constraint=gpytorch.constraints.Interval(
                lower_bound = lower,
                upper_bound = upper,
            )
        )
        matern.initialize(lengthscale=ls_init)

        self.covar_module = gpytorch.kernels.ScaleKernel(matern)
    @property
    def flow(self):
        return self._flow

    def forward(self, T, XY):
        A = self.flow(T, XY)
        TA = torch.cat([T, A], dim = -1)
        mean_x  = self.mean_module(TA)
        covar_x = self.covar_module(TA)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)