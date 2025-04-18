import torch
from typing import Any, Optional, Union
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from utils.plot import plot_scalar_field

class ScalarField:
    def __init__(
        self,
        times: torch.Tensor,             
        locations: torch.Tensor,         
        scalar: torch.Tensor            
    ) -> None:
        self.times = times
        self.locations = locations
        self.scalar = scalar
        self.map: Optional[Any] = None

    def plot(
        self,
        factor: int = 1,
        frame: int = 0,
        gif: bool = False
    ) -> Union[Figure, FuncAnimation]:
        fac = max(1, factor)
        scalar = self.scalar[:, ::fac, ::fac]  
        return plot_scalar_field(scalar, self.map, frame, gif)