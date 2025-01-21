import torch
import gpytorch
import net

class Mean(gpytorch.means.Mean):
    def __init__(self, flow, **kwargs):
        super(Mean, self).__init__(**kwargs)
        self.flow = flow
        self.Mean_W = net.Mean(L = 3)
        net.initialize(self.Mean_W)
        
    def forward(self, xyt, **params):
        a = torch.vmap(self.flow)(xyt)
        m = torch.vmap(self.Mean_W)(a).squeeze()
        return m

class Kernel(gpytorch.kernels.Kernel):
    def __init__(self, flow, **kwargs):
        super(Kernel, self).__init__(**kwargs)
        self.flow = flow
        self.Kernel_W = gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.MaternKernel(
                        nu = 1/2, ard_num_dims = 2))
    
    def forward(self, xyt1, xyt2, **params):
        t1 = xyt1[:,2:]
        t2 = xyt2[:,2:]
        
        a1 = torch.vmap(self.flow)(xyt1)
        a2 = torch.vmap(self.flow)(xyt2)
        
        #ta1 = torch.cat([t1,a1], dim = 1)
        #ta2 = torch.cat([t2,a2], dim = 1)
        
        return self.Kernel_W(a1, a2)
    
class GP(gpytorch.models.ExactGP):
    def __init__(self, xyt, z, flow, likelihood):
        super(GP, self).__init__(xyt, z, likelihood)
        self.Mean = Mean(flow)
        self.Kernel = Kernel(flow)
        
    def forward(self, xyt):
        mean = self.Mean(xyt)
        covar = self.Kernel(xyt, xyt)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class GP_FLOW:
    def __init__(self, flow, vel):
        self.flow = flow
        net.initialize(flow)

        self.vel = vel
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        xyt = None
        z = None
        
        self.gp = GP(xyt, z, flow, self.likelihood)