# src/fields/vector_field.py
import torch
import train.scale
import train.optim
import models.neural_flow
import models.gp
import utils.plot
import gpytorch

def linear_vector_field(TXY, a=1.0, b=1.0):
    XY = TXY[:, 1:3]
    A = b * torch.tensor([[0., -1.],
                          [1.,  0.]], device=XY.device, dtype=XY.dtype)
    v = XY @ A.T
    mean_norm = XY.norm(dim=1).mean()
    scale = a / (b * mean_norm + 1e-8)
    return v * scale

class VectorField:
    def __init__(self, velocity=None):
        self.velocity = velocity
        if velocity is None:
            self.velocity = models.neural_flow.Velocity()
            
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
    
    def train_mse_vector(self, vector_field, batch_size=None, epochs=100):
        TXY_scaled = vector_field.coord_field.TXY_scaled
        UV_scaled = vector_field.UV_scaled
        train.optim.mse_vector(TXY_scaled, UV_scaled, self.velocity, batch_size, epochs)
    
    def train_mle(self, scalar_field, batch_size=None, epochs=10):
        TXY_scaled = scalar_field.coord_field.TXY_scaled
        UV_scaled = linear_vector_field(TXY_scaled)
        train.optim.mse_vector(TXY_scaled, UV_scaled, self.velocity, 1000, 10)
        
        Z_scaled = scalar_field.Z_scaled
        self.flow = models.neural_flow.ODEFlow(self.velocity)
        self.gp = models.gp.TransportGP(self.flow)
        train.optim.mle(TXY_scaled, Z_scaled, self.gp, batch_size, epochs)
    
