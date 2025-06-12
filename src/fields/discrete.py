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
    def T_min(self):
        return self.TXY[:,0:1].min()
    
    @property
    def T_std(self):
        return self.TXY[:,0:1].std(unbiased=False)

    @property
    def XY_center(self):
        return self.TXY[:,1:3].mean(dim=0)
    
    @property
    def XY_std(self):
        return self.TXY[:,1:3].std(dim=0, unbiased=False)

    @property
    def T_scaled(self):
        return (self.TXY[:,0:1] - self.T_min) / self.T_std

    @property
    def XY_scaled(self):
        return (self.TXY[:,1:3] - self.XY_center) / self.XY_std
    
    @property
    def TXY_scaled(self):
        TXY_scaled = torch.cat([self.T_scaled, self.XY_scaled], dim=-1)
        return TXY_scaled

    def coarsen(self,factor=1):
        if self.grid is None:
            raise ValueError("coord_field.grid must not be None when using factor.")
        else:
            T = self.TXY[:,0:1]
            XY = self.TXY[:, 1:3]
            n, k1, k2 = self.grid
            fac = max(1, factor)
            
            T = T.reshape(n, k1, k2)[:, ::fac, ::fac]
            XY = XY.reshape(n, k1, k2, 2)[:, ::fac, ::fac, :]            
            self.grid = XY.shape[:3]
            
            T = T.reshape(-1, 1)
            XY = XY.reshape(-1, 2)
            self.TXY = torch.cat([T, XY], dim=-1)

class ScalarField:
    def __init__(self, coord_field, Z=None):
        self.coord_field = coord_field
        self.Z = Z
        
    @property
    def Z_mean(self):
        return self.Z.mean()

    @property
    def Z_std(self):
        return self.Z.std(unbiased=False)

    @property
    def Z_scaled(self):
        return (self.Z - self.Z_mean) / self.Z_std
    
    def simulate(self, flow):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        gp = models.gp.TransportGP(flow)
        gp.eval()
        gp.to(device)
        TXY_scaled = self.coord_field.TXY_scaled.to(device)
        with gpytorch.settings.prior_mode(True):
            self.Z = gp.likelihood(gp(TXY_scaled)).sample()
        
        self.Z = self.Z.to("cpu")
        del gp
        
    def coarsen(self, factor=1):
        if self.coord_field.grid is None:
            raise ValueError("coord_field.grid must not be None when using factor.")
        else:
            n, k1, k2 = self.coord_field.grid
            fac = max(1, factor)            
            self.Z = self.Z.reshape(n, k1, k2)[:, ::fac, ::fac].reshape(-1)
            self.coord_field.coarsen(factor)
            
    def plot(self):
        return utils.plot.scalar_field(self.coord_field.TXY, self.Z, self.coord_field.proj, extent=self.coord_field.extent)



class VectorField:
    def __init__(self, coord_field, UV):
        self.coord_field = coord_field
        self.UV = UV
        
    @property
    def RMS(self):
        norm_squared = torch.norm(self.UV, p = 2, dim=-1).square()        
        rms_per_timestep = norm_squared.mean(dim=-1).sqrt()         
        return rms_per_timestep.mean()
    
    @property
    def UV_scaled(self):
        return (self.coord_field.T_std / self.coord_field.XY_std) * self.UV

    def coarsen(self, factor):
        if self.coord_field.grid is None:
            raise ValueError("coord_field.grid must not be None when using factor.")
        else:
            n, k1, k2 = self.coord_field.grid
            fac = max(1, factor)      
            self.UV = self.UV.reshape(n, k1, k2, 2)[:, ::fac, ::fac, :].reshape(-1, 2)
            self.coord_field.coarsen(factor)

    def plot(self):
        return utils.plot.vector_field(self.coord_field.TXY, self.UV, self.coord_field.proj, extent=self.coord_field.extent)


