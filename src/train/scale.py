import torch
from typing import Callable

class NormalizeTime:
    def __init__(self, T: torch.Tensor) -> None:
        self.scale = T.max()
        
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t / self.scale.to(t.device)

class NormalizeLocation:
    def __init__(self, XY: torch.Tensor) -> None:
        xy_min = XY.min(dim=0).values        
        xy_max = XY.max(dim=0).values        
        self.center = (xy_min + xy_max) / 2  
        half_ranges = (xy_max - xy_min) / 2  
        self.scale = half_ranges.max()

    def __call__(self, xy: torch.Tensor) -> torch.Tensor:
        return (xy - self.center.to(xy.device)) / self.scale.to(xy.device)

class NormalizeLocationInverse:
    def __init__(self, XY: torch.Tensor) -> None:
        xy_min = XY.min(dim=0).values        
        xy_max = XY.max(dim=0).values       
        self.center = (xy_min + xy_max) / 2  
        half_ranges = (xy_max - xy_min) / 2  
        self.scale = half_ranges.max()      

    def __call__(self, xy: torch.Tensor) -> torch.Tensor:
        return xy * self.scale.to(xy.device) + self.center.to(xy.device)

class NormalizeScalar:
    def __init__(
        self,
        Z: torch.Tensor
    ) -> None:
        flat: torch.Tensor = Z.reshape(-1)
        self.mean: torch.Tensor = flat.mean()
        self.scale: torch.Tensor = flat.std()

    def __call__(
        self,
        z: torch.Tensor
    ) -> torch.Tensor:
        return (z -  self.mean.to(z.device)) / self.scale.to(z.device)

class ScaleFlow:
    def __init__(
        self,
        flow: Callable[[torch.Tensor], torch.Tensor],
        nt: NormalizeTime,
        nl: NormalizeLocation,
        nli: NormalizeLocationInverse,
    ) -> None:
        self.nt: NormalizeTime = nt
        self.nl: NormalizeLocation = nl
        self.nli: NormalizeLocationInverse = nli
        self.flow: Callable[[torch.Tensor], torch.Tensor] = flow

    def __call__(
        self,
        TXY: torch.Tensor
    ) -> torch.Tensor:
        T: torch.Tensor = self.nt(TXY[..., :1])
        XY: torch.Tensor = self.nl(TXY[..., 1:])
        TXY: torch.Tensor = torch.cat([T, XY], dim=-1)
        return self.nli(self.flow(TXY))
    
    
def rescale_temporal_lengthscale(nt, temporal_lengthscale):
    return temporal_lengthscale * nt.scale

def rescale_spatial_lengthscales(nl, spatial_lengthscales):
    return spatial_lengthscales * nl.scale

def rescale_variance(ns, variance_normalized):
    return variance_normalized * ns**2