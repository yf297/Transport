# src/fields/vector_field.py
import torch
import train.scale
import train.optim
import models.neural_flow
import models.gp
import utils.plot

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
        self.scaled_flow = models.neural_flow.NeuralFlow()

    def plot(
        self,
        coord_field,
        factor=1,
        frame=0,
        gif=False,
    ):
        T = coord_field.T
        n = T.size(0)
        XY = coord_field.XY
        if coord_field.grid is not None:
            k1 = coord_field.grid[0]
            k2 = coord_field.grid[1]
            fac = max(1, factor)

            XY = XY.reshape(k1, k2, 2)
            XY = XY[::fac, ::fac, :].reshape(-1,2)
        
        UV = self.velocity(T, XY).reshape(n, -1, 2) 
        return utils.plot.vector_field(
            XY=XY,
            UV=UV,
            proj=coord_field.proj,
            extent=coord_field.extent,
            frame=frame,
            gif=gif,
        )
        

        
    def train(
        self,
        scalar_field,
        epochs, 
        factor,
        step,
        size
    ):
    
        T = scalar_field.coord_field.T[::step]
        n = T.size(0)
        XY = scalar_field.coord_field.XY
        Z = scalar_field.Z[::step, :]
        
        if scalar_field.coord_field.grid is not None:
            k1 = scalar_field.coord_field.grid[0]
            k2 = scalar_field.coord_field.grid[1]
            fac = max(1, factor)
            
            XY = XY.reshape(k1, k2, 2)
            XY = XY[::fac, ::fac, :].reshape(-1,2)

            Z = Z.reshape(n, k1, k2)
            Z = Z[:, ::fac, ::fac].reshape(n, -1)
            
        scale_T = train.scale.ScaleT(T)
        scale_XY = train.scale.ScaleXY(XY)
        descale_A = train.scale.DeScaleA(scale_XY)
        scale_Z = train.scale.ScaleZ(Z)
        vecchia_blocks = train.vecchia.VecchiaBlocks(scale_T(T), scale_XY(XY), scale_Z(Z))        
        
        l_0 = 4*3600 / scale_T.scale
        l_1 = 200e3 / scale_XY.scale
        l_2 = 200e3 / scale_XY.scale
        
        self.gp = models.gp.GP(self.scaled_flow, l_0, l_1, l_2)
        T, XY, Z = vecchia_blocks.all()
        train.optim.mle(T, XY, Z, self.gp, epochs, size)
        self.flow = train.scale.DeScaleFlow(self.scaled_flow, scale_T, scale_XY, descale_A)
        
        def func(T, XY):
            T = T + 1e-8
            Jacobians = torch.vmap(torch.func.jacrev(self.flow, argnums=(0, 1)))(T, XY)
            Dt = Jacobians[0]
            Dx = Jacobians[1]
            vector = torch.linalg.solve(Dx, -Dt).squeeze(-1)        
            return vector
        
        def velocity(T, XY):
            n = T.size(0)
            m = XY.size(0)
            T = T.repeat_interleave(m).unsqueeze(1)
            XY = XY.repeat(n, 1)
            return func(T,XY).reshape(-1,2)
            
        self.velocity = velocity
        

    def RMS(
        self,
        coord_field
    ):
        rms= 0
        for frame in range(coord_field.T.size(0)):
            t = coord_field.T[frame].unsqueeze(0)
            XY = coord_field.XY
        
            UV = self.velocity(t, XY)
            U = UV[:,0]
            V = UV[:,1]
            rms +=round(torch.sqrt((U**2 + V**2).mean()).item(), 2) 
        return rms / coord_field.T.size(0)

    def RMSE(
        self,
        vector_field,
        frame,
    ):
        XY = vector_field.coord_field.XY
        t = vector_field.coord_field.T[frame].unsqueeze(0)
        
        UV0 = vector_field.UV[frame,:, :]
        UV1 = self.velocity(t, XY)
    
        U0 = UV0[:,0]
        V0 = UV0[:,1]
        U1 = UV1[:,0]  
        V1 = UV1[:,1]
        err = round(torch.sqrt(( (U0 - U1)**2 + (V0 - V1)**2).mean()).item(), 2)

        return err