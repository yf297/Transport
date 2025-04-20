import torch
from typing import Callable

class NormalizeTime:
    def __init__(
        self,
        T: torch.Tensor
    ) -> None:
        self.max: torch.Tensor = T.max()

    def __call__(
        self,
        t: torch.Tensor
    ) -> torch.Tensor:
        max_val: torch.Tensor = self.max.to(t.device)
        return t / max_val

class NormalizeLocation:
    def __init__(
        self,
        XY: torch.Tensor
    ) -> None:
        self.min: torch.Tensor = XY.min(dim=0).values
        self.max: torch.Tensor = XY.max(dim=0).values

    def __call__(
        self,
        xy: torch.Tensor
    ) -> torch.Tensor:
        min_val: torch.Tensor = self.min.to(xy.device)
        max_val: torch.Tensor = self.max.to(xy.device)
        return 2 * (xy - min_val) / (max_val - min_val) - 1

class NormalizeLocationInverse:
    def __init__(
        self,
        XY: torch.Tensor
    ) -> None:
        self.min: torch.Tensor = XY.min(dim=0).values
        self.max: torch.Tensor = XY.max(dim=0).values

    def __call__(
        self,
        xy: torch.Tensor
    ) -> torch.Tensor:
        min_val: torch.Tensor = self.min.to(xy.device)
        max_val: torch.Tensor = self.max.to(xy.device)
        return 0.5 * (xy + 1) * (max_val - min_val) + min_val

class NormalizeScalar:
    def __init__(
        self,
        Z: torch.Tensor
    ) -> None:
        flat: torch.Tensor = Z.reshape(-1)
        self.mean: torch.Tensor = flat.mean()
        self.std: torch.Tensor = flat.std()

    def __call__(
        self,
        z: torch.Tensor
    ) -> torch.Tensor:
        mean_val: torch.Tensor = self.mean.to(z.device)
        std_val: torch.Tensor = self.std.to(z.device)
        return (z - mean_val) / std_val

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
    
    
def rescale_temporal_lengthscale(T, temporal_lengthscale):
    return temporal_lengthscale *T.max()

def rescale_spatial_lengthscales(XY, spatial_lengthscales):
    min = XY.min(dim=0).values
    max = XY.max(dim=0).values
    return spatial_lengthscales * ( (max - min) / 2)

def rescale_variance(Z, variance_normalized):
    std_Z = Z.reshape(-1).std()
    return variance_normalized * std_Z**2