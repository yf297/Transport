# src/fields/vector_field.py
import torch
import train.scale
import train.optim
import models.neural_flow
import models.gp
import utils.plot
import torch.nn.functional as F
import gpytorch

class VectorField:
    def __init__(self):
        self.velocity = models.neural_flow.Velocity(width=64, depth=3)
        self.ode = models.neural_flow.ODE(self.velocity)
        self.gp = models.gp.TransportGP(self.ode)
    
    def plot(self, coord_field, factor=1):
        T = coord_field.TXY_scaled[:,0:1]
        XY = coord_field.TXY_scaled[:, 1:3]
        n, k1, k2 = coord_field.grid
        fac = max(1, factor)
        
        T = T.reshape(n, k1, k2)[:, ::fac, ::fac]
        XY = XY.reshape(n, k1, k2, 2)[:, ::fac, ::fac, :]            
            
        T = T.reshape(-1, 1)
        XY = XY.reshape(-1, 2)
        TXY_scaled = torch.cat([T, XY], dim=-1)
        UV = (coord_field.XY_scale/coord_field.T_scale) * self.velocity(TXY_scaled)
        
        T = coord_field.TXY[:,0:1]
        XY = coord_field.TXY[:, 1:3]
        n, k1, k2 = coord_field.grid
        fac = max(1, factor)
        
        T = T.reshape(n, k1, k2)[:, ::fac, ::fac]
        XY = XY.reshape(n, k1, k2, 2)[:, ::fac, ::fac, :]            
            
        T = T.reshape(-1, 1)
        XY = XY.reshape(-1, 2)
        TXY = torch.cat([T, XY], dim=-1)
        return utils.plot.vector_field(TXY, UV, coord_field.proj, coord_field.extent)
        
        
    def train_gp(self, dvf, scalar_field, batch_size=1000, epochs=10):
        TXY_scaled = scalar_field.coord_field.TXY_scaled
        Z_scaled = scalar_field.Z_scaled
        
        T = TXY_scaled[..., 0]  
        idx = torch.nonzero(T == 0.0, as_tuple=True)[0]  
        
        TXY_scaled = TXY_scaled[idx]
        Z_scaled = Z_scaled[idx]
        train.optim.mle(TXY_scaled, Z_scaled, self.gp, batch_size, epochs, True, self, dvf)

    def train_flow(self, dvf, scalar_field, batch_size=1000, epochs=10):
        TXY_scaled = scalar_field.coord_field.TXY_scaled
        Z_scaled = scalar_field.Z_scaled
        
        mean = self.gp.mean.constant.item()
        sigma2 = self.gp.kernel.kernel.outputscale.item()
        ls = self.gp.kernel.kernel.base_kernel.lengthscale[0].detach()
        tau2 = self.gp.likelihood.noise.item()
        
        upper_out = sigma2 + 0.0001
        lower_out = sigma2 - 0.0001
        
        upper_noise = tau2 + 0.00001
        lower_noise = tau2 - 0.00001
        
        upper_mean = mean + 0.0001
        lower_mean = mean - 0.0001
        
        upper_ls = ls + torch.tensor([0.01, 0.0001, 0.0001]) 
        lower_ls = ls - torch.tensor([0.01, 0.0001, 0.0001]) 
        
        self.gp.kernel.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.MaternKernel(
            nu=1.5,
            ard_num_dims=3,
            lengthscale_constraint=gpytorch.constraints.Interval(
                lower_bound=lower_ls,
                upper_bound=upper_ls,
            )),  outputscale_constraint=gpytorch.constraints.Interval(lower_bound=lower_out, upper_bound=upper_out)
        )
        self.gp.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.Interval(
                lower_noise, upper_noise
            )
        )
        self.gp.mean = gpytorch.means.ConstantMean(
            constant_constraint=gpytorch.constraints.Interval(
               lower_mean, upper_mean
            )
        )
        train.optim.mle(TXY_scaled, Z_scaled, self.gp, batch_size, epochs, True, self, dvf)

    def RMSE(self, vector_field):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        UV_true = vector_field.UV.to(device)
        TXY_scaled = vector_field.coord_field.TXY_scaled.to(device)
        UV_hat = (vector_field.coord_field.XY_scale/vector_field.coord_field.T_scale) * self.velocity(TXY_scaled)
        UV = UV_true - UV_hat
        n = vector_field.coord_field.grid[0]
        return UV.reshape(n, -1, 2).norm(p = 2, dim=-1).square().mean(dim=-1).sqrt().mean()

    
    def RMS(self, coord_field):
        n = coord_field.grid[0]
        TXY_scaled = coord_field.TXY_scaled
        UV_hat = (coord_field.XY_scale/coord_field.T_scale) * self.velocity(TXY_scaled)
        return UV_hat.reshape(n, -1, 2).norm(p = 2, dim=-1).square().mean(dim=-1).sqrt().mean()
