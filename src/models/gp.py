# src/models/gp_flow.py
import torch
import gpytorch.likelihoods
import gpytorch.means
import gpytorch.kernels
import gpytorch.distributions

class GP(gpytorch.models.ExactGP):
    def __init__(
        self,
        flow: torch.nn.Module
    ):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(None, None, likelihood)
        self.flow = flow
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.keops.MaternKernel(nu=2.5, ard_num_dims=3)
        )
        self.covar_module.base_kernel.initialize(
            lengthscale=torch.tensor([6.0, 0.3, 0.3]))

    def forward(self, 
                TXY: torch.Tensor
    ) -> gpytorch.distributions.MultivariateNormal:
        T = TXY[..., 0:1]
        TA = torch.cat([T, self.flow(TXY)], dim = -1)  
        mean_x = self.mean_module(TA)
        covar_x = self.covar_module(TA)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)