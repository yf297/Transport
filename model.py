
import torch
import gpytorch
import net

class Mean(gpytorch.means.Mean):
    def __init__(self, **kwargs):
        super(Mean, self).__init__(**kwargs)
        self.func = net.Scalar_Field(L = 4)
        net.initialize(self.func)
        
    def forward(self, a, **params):
        m = torch.vmap(self.func)(a).squeeze()
        return m


class GP(gpytorch.models.ExactGP):
    def __init__(self, tx, y, Flow, Vel, likelihood):
        super(GP, self).__init__(tx, y, likelihood)
        self.Flow = Flow
        self.Vel = Vel
        self.Mean_Initial = Mean()
        self.Kernel_Initial = gpytorch.kernels.ScaleKernel(
                              gpytorch.kernels.MaternKernel(
                                nu = 3/2, 
                                ard_num_dims = 2))
        
    def forward(self, tx):
        a = torch.vmap(self.Flow)(tx)
        
        mean = self.Mean_Initial(a)
        covar = self.Kernel_Initial(a, a)
       
        return gpytorch.distributions.MultivariateNormal(mean, covar)

def Initialize(Flow, Vel):
    
    net.initialize(Flow)
    net.initialize(Vel)
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    tx = None
    y = None
    
    return GP(tx, y, Flow, Vel, likelihood)