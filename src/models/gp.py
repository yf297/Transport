# src/models/gp_flow.py
from typing import Callable
import torch
from torch import Tensor
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, keops

class GP(ExactGP):
    def __init__(
        self,
        flow: Callable[[Tensor], Tensor]
    ) -> None:
        likelihood = GaussianLikelihood()
        super().__init__(None, None, likelihood)
        self.flow = flow
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            keops.MaternKernel(nu=2.5, ard_num_dims=3)
        )
        self.covar_module.base_kernel.initialize(
            lengthscale=torch.tensor([6.0, 0.1, 0.1]))

    def forward(self, TXY: Tensor) -> MultivariateNormal:
        T = TXY[..., 0:1]
        TA = torch.cat([T, self.flow(TXY)], dim = -1)  
        mean_x  = self.mean_module(TA)
        covar_x = self.covar_module(TA)
        return MultivariateNormal(mean_x, covar_x)
