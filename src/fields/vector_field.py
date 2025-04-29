import torch
import matplotlib.figure
import matplotlib.animation
import fields.coord_field
import fields.scalar_field
import train.scale
import train.optim
import models.neural_flow
import models.gp
import utils.plot
from typing import Union, Callable, Optional

class DiscreteVectorField:
    def __init__(
        self,
        coord_field: fields.coord_field.DiscreteCoordField,
        UV: torch.Tensor,
    ):
        self.coord_field = coord_field
        self.UV = UV
        self.n = self.UV.size(0)

    def plot(
        self,
        factor: int = 1,
        frame: int = 0,
        gif: bool = False,
    ) -> Union[matplotlib.figure.Figure, matplotlib.animation.FuncAnimation]:
        
        UV = self.UV 
        XY = self.coord_field.XY
        if self.coord_field.grid is not None:
            fac = max(1, factor)
            UV = self.UV.reshape(self.n,
                        self.coord_field.grid[0], 
                        self.coord_field.grid[1], 2)
            UV = UV[:, ::fac, ::fac, :].reshape(self.n, -1, 2)
            
            XY = self.coord_field.XY.reshape(self.coord_field.grid[0],
                        self.coord_field.grid[1], 2)
            XY = XY[::fac, ::fac, :].reshape(-1,2)
        
        return utils.plot.vector_field(
            XY=XY,
            UV=UV,
            proj=self.coord_field.proj,
            extent=self.coord_field.extent,
            frame=frame,
            gif=gif,
        )
        
    def RMS(
        self, 
        frame: int
    ) -> float:
        UV = self.UV[frame,:, :]
        U = UV[:,0]
        V = UV[:,1]
        return round(torch.sqrt((U**2 + V**2).mean()).item(), 2)
        

class ContinuousVectorField:
    def __init__(
        self,
        func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ):
        self.func = func
    
    def plot(
        self,
        coord_field: fields.coord_field.DiscreteCoordField,
        factor: int = 1,
        frame: int = 0,
        gif: bool = False,
    ) -> Union[matplotlib.figure.Figure, matplotlib.animation.FuncAnimation]:
       
        T = coord_field.T
        XY = coord_field.XY
        if coord_field.grid is not None:
            fac = max(1, factor)  
            XY = coord_field.XY.reshape(coord_field.grid[0],
                        coord_field.grid[1], 2)
            XY = XY[::fac, ::fac, :].reshape(-1,2)
       
        UV = self.func(T, XY).reshape(T.size(0), -1, 2)
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
        scalar_field: fields.scalar_field.DiscreteScalarField,
        epochs: int, 
        nn: int,
        k: int,
        size: int
        ) -> None:
    
        T = scalar_field.coord_field.T
        XY = scalar_field.coord_field.XY
        Z = scalar_field.Z
       
        nt = train.scale.NormalizeTime(T)
        nl = train.scale.NormalizeLocation(XY)
        nli = train.scale.NormalizeLocationInverse(XY)
        ns = train.scale.NormalizeScalar(Z)
    
        flow = models.neural_flow.NeuralFlow()
        gp = models.gp.GP(flow)
        
        ell_phys = 80e3
        tau_phys = 24*3600     
        ls = nl.scale   
        ts = nt.scale

        ell_x = ell_phys / ls
        ell_y = ell_phys / ls
        ell_t = tau_phys / ts

        gp.covar_module.base_kernel.initialize(
            lengthscale=torch.tensor([ell_t, ell_x, ell_y])
        )
        train.optim.mle(nt(T), nl(XY), ns(Z), gp, epochs, nn, k, size)
        
        def func0(TXY):
            Jacobians = torch.vmap(torch.func.jacrev(train.scale.ScaleFlow(gp.flow,nt,nl,nli)))(TXY)
            Dt = Jacobians[..., 0:1]
            Dx = Jacobians[..., 1:]
            vector = torch.linalg.solve(Dx, -Dt).squeeze(-1)        
            return vector
        
        def func(T, XY):
            TXY = torch.cat([
            T.repeat_interleave(XY.size(0)).unsqueeze(1),
            XY.repeat(T.size(0), 1)
            ], dim=-1)

            return func0(TXY).reshape(-1,2)
            
        self.func = func
        self.sigma2 =  round(gp.covar_module.outputscale.item() * nt.scale.item(), 2)
        self.l0  = round(gp.covar_module.base_kernel.lengthscale[0][0].item() * nt.scale.item(), 2)
        self.l1 = round(gp.covar_module.base_kernel.lengthscale[0][1:][0].item() * nl.scale.item(), 2)
        self.l2 = round(gp.covar_module.base_kernel.lengthscale[0][1:][1].item() * nl.scale.item(), 2)
        self.tau2 =  round(gp.likelihood.noise.item(),2)
        

    def RMSE(
        self,
        vector_field: DiscreteVectorField,
        frame: int ) -> float:
        T = vector_field.coord_field.T[frame:(frame+1)]
        XY = vector_field.coord_field.XY
        
        UV0 = vector_field.UV[frame,:, :]
        UV1 = self.func(T, XY)
       
        U0 = UV0[:,0]
        V0 = UV0[:,1]
        U1 = UV1[:,0]  
        V1 = UV1[:,1]
        
        return round(torch.sqrt(( (U0 - U1)**2 + (V0 - V1)**2).mean()).item(), 2)
