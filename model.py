
import torch
import gpytorch
import net

class Mean(gpytorch.means.Mean):
    def __init__(self, **kwargs):
        super(Mean, self).__init__(**kwargs)
        self.Mean_W = net.Mean(L = 4)
        net.initialize(self.Mean_W)
        
    def forward(self, a, **params):
        m = torch.vmap(self.Mean_W)(a).squeeze()
        return m

class Kernel(gpytorch.kernels.Kernel):
    def __init__(self, **kwargs):
        super(Kernel, self).__init__(**kwargs)
        self.Kernel_W = gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.MaternKernel(
                        nu = 3/2, ard_num_dims = 2))
    
    def forward(self, a1, a2, **params):
        return self.Kernel_W(a1, a2)
    
class GP(gpytorch.models.ExactGP):
    def __init__(self, tx, y, Flow, Vel, likelihood):
        super(GP, self).__init__(tx, y, likelihood)
        self.Flow = Flow
        self.Vel = Vel
        self.Mean = Mean()
        self.Kernel = Kernel()

    def forward(self, tx):
        a = torch.vmap(self.Flow)(tx)
        mean = self.Mean(a)
        covar = self.Kernel(a, a)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

def Initialize(Flow, Vel):
    
    net.initialize(Flow)
    net.initialize(Vel)
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    tx = None
    y = None
    
    return GP(tx, y, Flow, Vel, likelihood)