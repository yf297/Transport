import torch
import cartopy.crs
from typing import Tuple, Optional

class DiscreteCoordField:
    def __init__(
        self,
        T: torch.Tensor,
        XY: torch.Tensor,
        proj: Optional[cartopy.crs.CRS] = None,
        extent: Optional[Tuple[float, float, float, float]] = None,
        grid: Optional[Tuple[int, int]] = None,
    ):
        self.T = T
        self.XY = XY
        self.proj = proj
        self.extent = extent
        self.grid = grid