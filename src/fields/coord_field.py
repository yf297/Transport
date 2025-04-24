import torch
import cartopy.crs
from typing import Tuple, Optional

class DiscreteCoordField:
    def __init__(
        self,
        times: torch.Tensor,
        locations: torch.Tensor,
        proj: Optional[cartopy.crs.CRS] = None,
        extent: Optional[Tuple[float, float, float, float]] = None,
    ):
        self.times = times
        self.locations = locations
        self.proj = proj
        self.extent = extent