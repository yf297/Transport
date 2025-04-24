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
        vector: torch.Tensor,
    ):
        self.coord_field = coord_field
        self.vector = vector

    def plot(
        self,
        factor: int = 1,
        frame: int = 0,
        gif: bool = False,
    ) -> Union[matplotlib.figure.Figure, matplotlib.animation.FuncAnimation]:
        fac = max(1, factor)
        vector = self.vector[:, ::fac, ::fac, :]
        locations = self.coord_field.locations[::fac, ::fac, :]
        
        return utils.plot.vector_field(
            locations=locations,
            vector=vector,
            proj=self.coord_field.proj,
            extent=self.coord_field.extent,
            frame=frame,
            gif=gif,
        )
        
    def RMS(
        self, 
        frame: int
    ) -> float:
        locations = self.coord_field.locations
        vector0 = self.vector[frame,:,:,:]
        
        return round(torch.sqrt( ((vector0[:,:,0])**2 + (vector0[:,:,1] )**2).mean() ).item(), 2)

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
        
        fac = max(1, factor)
        locations = coord_field.locations[::fac, ::fac, :]
        H, W, _ = locations.shape
        times = coord_field.times
       
        vector = self.func(times, locations)

        return utils.plot.vector_field(
            locations=locations,
            vector=vector,
            proj=coord_field.proj,
            extent=coord_field.extent,
            frame=frame,
            gif=gif,
        )
        
    def train(
        self,
        scalar_field: fields.scalar_field.DiscreteScalarField,
        epochs: int = 50, 
        nn: int = 1,
        k: int = 4,
        size: int = 1000 
        ) -> None:
    
        T = scalar_field.coord_field.times
        XY = scalar_field.coord_field.locations.reshape(-1, 2)
        Z = scalar_field.scalar.reshape(T.size(0), -1)
       
        nt = train.scale.NormalizeTime(T)
        nl = train.scale.NormalizeLocation(XY)
        nli = train.scale.NormalizeLocationInverse(XY)
        ns = train.scale.NormalizeScalar(Z)
    
        flow = models.neural_flow.NeuralFlow()
        gp = models.gp.GP(flow)
        
        ell_phys = 100e3
        tau_phys = 24*3600     
        ls = nl.scale   
        ts = nt.scale

        ell_x = ell_phys / ls
        ell_y = ell_phys / ls
        ell_t = tau_phys / ts
        #gp.likelihood.noise_covar.initialize(noise=torch.tensor(0.1))
        gp.covar_module.base_kernel.initialize(
            lengthscale=torch.tensor([ell_t, ell_x, ell_y]))
        train.optim.mle(nt(T), nl(XY), ns(Z), gp, epochs, nn, k, size)

        def func0(TXY):
            Jacobians = torch.vmap(torch.func.jacrev(train.scale.ScaleFlow(gp.flow,nt,nl,nli)))(TXY)
            Dt = Jacobians[..., 0:1]
            Dx = Jacobians[..., 1:]
            vector = torch.linalg.solve(Dx, -Dt).squeeze(-1)        
            return vector
        
        def func(times, locations):
            H, W, _ = locations.shape
            T = times
            XY = locations.reshape(-1, 2)
            TXY = torch.cat([
            T.repeat_interleave(XY.size(0)).unsqueeze(1),
            XY.repeat(T.size(0), 1)
            ], dim=-1)

            return func0(TXY).reshape(T.size(0), H, W, 2)
            
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
        locations = vector_field.coord_field.locations
        times = vector_field.coord_field.times[frame:frame+1]
      
        vector1 = self.func(times, locations)[0,:,:,:]
        vector0 = vector_field.vector[frame,:,:,:]
        
        return round(torch.sqrt( ((vector1[:,:,0] - vector0[:,:,0])**2 + (vector1[:,:,1] - vector0[:,:,1] )**2).mean() ).item(), 2)