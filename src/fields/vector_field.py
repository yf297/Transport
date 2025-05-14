# src/fields/vector_field.py
import torch
import numpy as np
import train.scale
import train.optim
import models.neural_flow
import models.gp
import utils.plot
import torch.nn as nn
import math

class DiscreteVectorField:
    def __init__(
        self,
        coord_field,
        UV,
    ):
        self.coord_field = coord_field
        self.UV = UV

    def plot(
        self,
        factor=1,
        frame=0,
        gif=False,
        scale = 2e-4
    ):
        
        UV = self.UV
        n = UV.size(0)
        XY = self.coord_field.XY
        if self.coord_field.grid is not None:
            k1 = self.coord_field.grid[0]
            k2 = self.coord_field.grid[1]
            fac = max(1, factor)

            XY = XY.reshape(k1, k2, 2)
            XY = XY[::fac, ::fac, :].reshape(-1,2)
            
            UV = self.UV.reshape(n, k1, k2, 2)
            UV = UV[:, ::fac, ::fac, :].reshape(n, -1, 2)
            
        return utils.plot.vector_field(
            XY=XY,
            UV=UV,
            proj=self.coord_field.proj,
            extent=self.coord_field.extent,
            frame=frame,
            gif=gif,
            scale=scale
        )
        
    def RMS(
        self 
    ):
        rms = 0
        for frame in range(self.UV.size(0)):
            UV = self.UV[frame,:, :]
            U = UV[:,0]
            V = UV[:,1]
            rms +=round(torch.sqrt((U**2 + V**2).mean()).item(), 2) 
        return rms / self.UV.size(0)
        
        
        

class ContinuousVectorField:
    def __init__(
        self,
        velocity=None
    ):
        self.velocity = velocity

    def plot(self,
             coord_field, 
             factor=1, 
             frame=0,
             gif=False, 
             scale = 2e-4):
        
        T = coord_field.T
        n = T.size(0)
        XY = coord_field.XY

        if coord_field.grid is not None:
            k1, k2 = coord_field.grid
            fac = max(1, factor)
            XY = XY.reshape(k1, k2, 2)[::fac, ::fac, :].reshape(-1, 2)

        if coord_field.proj != self.proj:
            x = XY[:, 0].detach().cpu().numpy().reshape(-1, 1)
            y = XY[:, 1].detach().cpu().numpy().reshape(-1, 1)
            coords = self.proj.transform_points(coord_field.proj, x, y)
            XY0 = torch.from_numpy(coords[:, 0, 0:2]).float()
        else:
            XY0 = XY

        UV0 = torch.stack([self.velocity(t, XY0).detach() for t in T], dim = 0).reshape(-1,2)

        XY0 = XY0.repeat(n, 1)
        if coord_field.proj != self.proj:
            x0 = XY0[:, 0].detach().cpu().numpy().reshape(-1, 1)
            y0 = XY0[:, 1].detach().cpu().numpy().reshape(-1, 1)
            u0 = UV0[:, 0].cpu().numpy().reshape(-1, 1)
            v0 = UV0[:, 1].cpu().numpy().reshape(-1, 1)
            u, v = coord_field.proj.transform_vectors(self.proj, x0, y0, u0, v0)
            U = torch.from_numpy(u).squeeze(1)
            V = torch.from_numpy(v).squeeze(1)
        else:
            U = UV0[:, 0]
            V = UV0[:, 1]

        UV = torch.stack([U, V], dim=-1).reshape(n, -1, 2)

        return utils.plot.vector_field(
            XY=XY,
            UV=UV,
            proj=coord_field.proj,
            extent=coord_field.extent,
            frame=frame,
            gif=gif,
            scale=scale
        )


        
    def train(
        self,
        scalar_field,
        epochs=100, 
        factor=1,
        step=1,
        size=1000
    ):
    
        
        k1 = scalar_field.coord_field.grid[0]
        k2 = scalar_field.coord_field.grid[1]
        
        fac = max(1, factor)
        n = scalar_field.Z.size(0)
        XYg = scalar_field.coord_field.XY.reshape(k1, k2, 2)
        Zg = scalar_field.Z.reshape(n, k1, k2)
        
        T = scalar_field.coord_field.T[::step]
        XY = XYg[::fac, ::fac, :].reshape(-1,2)
        Z = Zg[::step, ::fac, ::fac].reshape(T.size(0), -1)

        scale_T = train.scale.ScaleT(T)
        scale_XY = train.scale.ScaleXY(XY)
        descale_A = train.scale.DeScaleA(scale_XY)
        scale_Z = train.scale.ScaleZ(Z)
           
        self.gp = models.gp.GP()
        self.scaled_flow = models.neural_flow.NeuralFlow()
        self.l0_raw = nn.Parameter(2*torch.tensor(math.log(math.exp(1.0) - 1.0)))

        train.optim.mle(scale_T(T), 
                        scale_XY(XY), 
                        scale_Z(Z), 
                        self.gp, 
                        self.scaled_flow, 
                        epochs, 
                        size)
        self.flow = train.scale.DeScaleFlow(self.scaled_flow, scale_T, scale_XY, descale_A)
        
        def velocity(t, XY):
            Dt, Dx = torch.vmap(torch.func.jacrev(self.flow, argnums=(0, 1)), in_dims=(None, 0))(t, XY)
            vector = torch.linalg.solve(Dx, -Dt).squeeze(-1)        
            return vector
            
        self.velocity = velocity
        self.proj = scalar_field.coord_field.proj
        

                    
    def RMS(self, coord_field):
        XY = coord_field.XY

        if coord_field.proj != self.proj:
            x = XY[:, 0].detach().cpu().numpy().reshape(-1, 1)
            y = XY[:, 1].detach().cpu().numpy().reshape(-1, 1)
            coords = self.proj.transform_points(coord_field.proj, x, y)
            XY0 = torch.from_numpy(coords[:, 0, 0:2]).float()
            x0 = XY0[:, 0].detach().cpu().numpy().reshape(-1, 1)
            y0 = XY0[:, 1].detach().cpu().numpy().reshape(-1, 1)
        else:
            XY0 = XY
            x0 = XY0[:, 0].detach().cpu().numpy().reshape(-1, 1)
            y0 = XY0[:, 1].detach().cpu().numpy().reshape(-1, 1)

        rms = 0.0
        for frame in range(coord_field.T.size(0)):
            t = coord_field.T[frame].unsqueeze(0)
            UV0 = self.velocity(t, XY0).detach()

            u, v = UV0[:, 0].cpu().numpy().reshape(-1, 1), UV0[:, 1].cpu().numpy().reshape(-1, 1)

            if coord_field.proj != self.proj:
                u, v = coord_field.proj.transform_vectors(self.proj, x0, y0, u, v)

            rms += np.sqrt((u**2 + v**2).mean()).item()

        return round(rms / coord_field.T.size(0), 2)

    
    
    def RMSE(self, vector_field, frame):
        
        t = vector_field.coord_field.T[frame].unsqueeze(0)

        XY = vector_field.coord_field.XY
        UV = vector_field.UV[frame,:,:] 

        if vector_field.coord_field.proj != self.proj:
            x, y = XY[:, 0], XY[:, 1]
            coords = self.proj.transform_points(
                vector_field.coord_field.proj,
                x.detach().cpu().numpy().reshape(-1, 1),
                y.detach().cpu().numpy().reshape(-1, 1)
            )
            XY0 = torch.from_numpy(coords[:, 0, 0:2]).float()
        else:
            XY0 = XY

        UV0 = self.velocity(t, XY0)
        UV0 = UV0.detach()

        if vector_field.coord_field.proj != self.proj:
            x0, y0 = XY0[:, 0], XY0[:, 1]
            u_hat, v_hat = UV0[:, 0], UV0[:, 1]
            u_hat, v_hat = vector_field.coord_field.proj.transform_vectors(
                self.proj,
                x0.detach().cpu().numpy().reshape(-1, 1),
                y0.detach().cpu().numpy().reshape(-1, 1),
                u_hat.cpu().numpy().reshape(-1, 1),
                v_hat.cpu().numpy().reshape(-1, 1),
            )
            u_hat = torch.from_numpy(u_hat).squeeze()
            v_hat = torch.from_numpy(v_hat).squeeze()
        else:
            u_hat, v_hat = UV0[:, 0], UV0[:, 1]

        err = torch.sqrt(((UV[:, 0] - u_hat) ** 2 + (UV[:, 1] - v_hat) ** 2).mean())
        return round(err.item(), 2)
