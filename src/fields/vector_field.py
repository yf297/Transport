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
        self 
    ) -> float:
        vector0 = self.vector
        return torch.sqrt(((vector0)**2).mean(dim=(1,2))).mean()


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
        sample_size: int = 1000,
        nn: int = 1,
        exact: bool = False,
        ) -> None:
    
        nt = train.scale.NormalizeTime(scalar_field.coord_field.times)
        nl = train.scale.NormalizeLocation(scalar_field.coord_field.locations.reshape(-1, 2))
        nli = train.scale.NormalizeLocationInverse(scalar_field.coord_field.locations.reshape(-1, 2))
        ns = train.scale.NormalizeScalar(scalar_field.scalar.reshape(scalar_field.coord_field.times.size(0), -1))
    
        T = scalar_field.coord_field.times
        XY = scalar_field.coord_field.locations.reshape(-1, 2)
        Z = scalar_field.scalar.reshape(T.size(0), -1)
        
        flow = models.neural_flow.NeuralFlow()
        gp = models.gp.GP(flow)
        #train.optim.mle0(nt(T), nl(XY), ns(Z), gp, epochs, sample_size)
        train.optim.mle(nt(T), nl(XY), ns(Z), gp, epochs, sample_size, nn, exact)

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
        self.sigma2 = train.scale.rescale_variance(Z, gp.covar_module.outputscale).item()
        self.l0 = train.scale.rescale_temporal_lengthscale(T,gp.covar_module.base_kernel.lengthscale[0][0]).item()
        self.l1 = train.scale.rescale_spatial_lengthscales(XY,gp.covar_module.base_kernel.lengthscale[0][1:])[0].item()
        self.l2 = train.scale.rescale_spatial_lengthscales(XY,gp.covar_module.base_kernel.lengthscale[0][1:])[1].item()

    def RMSE(
        self,
        vector_field: DiscreteVectorField
    ) -> float:
        locations = vector_field.coord_field.locations
        times = vector_field.coord_field.times
      
        vector1 = self.func(times, locations)
        vector0 = vector_field.vector
        
        return torch.sqrt(((vector1 - vector0)**2).mean(dim=(1,2))).mean()