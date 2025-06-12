# src/fields/vector_field.py
import torch
import train.scale
import train.optim
import models.neural_flow
import models.gp
import utils.plot
import gpytorch

class VectorField:
    def __init__(self, velocity=None):
        self.velocity = velocity


    def plot(self, coord_field):
        TXY_scaled = coord_field.TXY_scaled
        UV = (coord_field.XY_std/coord_field.T_std) * self.velocity(TXY_scaled)

        return utils.plot.vector_field(coord_field.TXY, UV, coord_field.proj, coord_field.extent)

        
    def train_mse(self, vector_field, batch_size=None, epochs=100):
        TXY_scaled = vector_field.coord_field.TXY_scaled
        UV_scaled = vector_field.UV_scaled
        self.velocity = models.neural_flow.Velocity()
        train.optim.mse(TXY_scaled, UV_scaled, self.velocity, batch_size, epochs)
        
    def train_mle(self, scalar_field, batch_size=None, epochs=100):
        TXY_scaled = scalar_field.coord_field.TXY_scaled
        Z_scaled = scalar_field.Z_scaled
        
        self.velocity = models.neural_flow.Velocity()
        self.flow = models.neural_flow.ODE(self.velocity)
        self.gp = models.gp.TransportGP(self.flow)

        train.optim.mle(TXY_scaled, Z_scaled, self.gp, batch_size, epochs)
