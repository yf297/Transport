# src/fields/discrete.py
import torch
import utils.plot
import gpytorch
import models.gp

class CoordField:
    def __init__(self, TXY, proj=None, extent=None, grid=None):
        self.TXY = TXY
        self.proj = proj
        self.extent = extent
        self.grid = grid
        
    @property
    def T_center(self):
        return self.TXY[:,0:1].min()
    
    @property
    def T_scale(self):
        return self.TXY[:,0:1].max() 

    @property
    def XY_center(self):
        return self.TXY[:,1:3].mean(dim=0)
    
    @property
    def XY_scale(self):
        return self.T_scale * 40

    @property
    def TXY_scaled(self):
        T_scaled = (self.TXY[:,0:1] - self.T_center) / self.T_scale
        XY_scaled =  (self.TXY[:,1:3] - self.XY_center) / self.XY_scale
        TXY_scaled = torch.cat([T_scaled, XY_scaled], dim=-1)
        return TXY_scaled
    

class ScalarField:
    def __init__(self, coord_field, Z=None):
        self.coord_field = coord_field
        self.Z = Z
        
    @property
    def Z_mean(self):
        return self.Z.mean()

    @property
    def Z_scale(self):
        return self.Z.std(unbiased=False)

    @property
    def Z_scaled(self):
        return (self.Z - self.Z_mean) / self.Z_scale
    
    
    def simulate_from_prior(self, flow = None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        gp = models.gp.TransportGP(flow)
        gp.likelihood.noise = torch.tensor(0.001)
        gp.to(device)
        gp.eval()
        TXY_scaled = self.coord_field.TXY_scaled.to(device)
        with gpytorch.settings.fast_computations(log_prob=False,
                                            covar_root_decomposition=False,
                                            solves=False):
            with gpytorch.settings.prior_mode(True):
                Z = gp(TXY_scaled).sample()
        gp.to("cpu")
        return Z.to("cpu")

    def plot(self):
        return utils.plot.scalar_field(self.coord_field.TXY, self.Z_scaled, self.coord_field.proj, extent=self.coord_field.extent)


class VectorField:
    def __init__(self, coord_field, UV):
        self.coord_field = coord_field
        self.UV = UV
        
    @property
    def RMS(self):
        n = self.coord_field.grid[0]
        return self.UV.reshape(n, -1, 2).norm(p = 2, dim=-1).square().mean(dim=-1).sqrt().mean()
    
    @property
    def RMS_scaled(self):
        n = self.coord_field.grid[0]
        return self.UV_scaled.reshape(n, -1, 2).norm(p = 2, dim=-1).square().mean(dim=-1).sqrt().mean()
    
    @property
    def UV_scaled(self):
        return (self.coord_field.T_scale / self.coord_field.XY_scale) * self.UV

    def plot(self, factor = 1):
        T = self.coord_field.TXY[:,0:1]
        XY = self.coord_field.TXY[:, 1:3]
        n, k1, k2 = self.coord_field.grid
        fac = max(1, factor)
        
        T = T.reshape(n, k1, k2)[:, ::fac, ::fac]
        XY = XY.reshape(n, k1, k2, 2)[:, ::fac, ::fac, :]            
            
        T = T.reshape(-1, 1)
        XY = XY.reshape(-1, 2)
        TXY = torch.cat([T, XY], dim=-1)
        UV = self.UV.reshape(n, k1, k2, 2)[:, ::fac, ::fac, :].reshape(-1, 2)
        return utils.plot.vector_field(TXY, UV, self.coord_field.proj, extent=self.coord_field.extent)

