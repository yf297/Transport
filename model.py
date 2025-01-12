
import torch
import gpytorch
import net

class Mean(gpytorch.means.Mean):
    def __init__(self, Flow, **kwargs):
        super(Mean, self).__init__(**kwargs)
        self.Flow = Flow
        self.Mean_W = net.Mean(L = 4)
        net.initialize(self.Mean_W)
        
    def forward(self, tx, **params):
        a = torch.vmap(self.Flow)(tx)
        m = torch.vmap(self.Mean_W)(a).squeeze()
        return m

class Kernel(gpytorch.kernels.Kernel):
    def __init__(self, Flow, **kwargs):
        super(Kernel, self).__init__(**kwargs)
        self.Flow = Flow
        self.Kernel_W = gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.MaternKernel(
                        nu = 3/2, ard_num_dims = 2))
    
    def forward(self, tx1, tx2, **params):

        a1 = torch.vmap(self.Flow)(tx1)
        a2 = torch.vmap(self.Flow)(tx2)

        return self.Kernel_W(a1, a2) 
    
class GP(gpytorch.models.ExactGP):
    def __init__(self, tx, y, Flow, Vel, likelihood):
        super(GP, self).__init__(tx, y, likelihood)
        self.Flow = Flow
        self.Vel = Vel
        self.Mean = Mean(self.Flow)
        self.Kernel = Kernel(self.Flow)

    def forward(self, x):
        mean = self.Mean(x)
        covar = self.Kernel(x, x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class Initialize:
    def __init__(self, Flow, Vel):
        self.Flow = Flow
        net.initialize(self.Flow)
        
        self.Vel = Vel
        net.initialize(self.Vel)
        
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        tx = None
        y = None
        
        self.GP = GP(tx, y, Flow, Vel, likelihood)