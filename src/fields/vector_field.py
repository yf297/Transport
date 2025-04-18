import torch
from typing import Any, Optional, Union
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from utils.plot import plot_vector_field

class VectorField:
    def __init__(
        self,
        times: torch.Tensor,             
        locations: torch.Tensor,        
        vector: torch.Tensor            
    ) -> None:
        self.times = times
        self.locations = locations
        self.vector = vector
        self.map: Optional[Any] = None

    def plot(
        self,
        factor: int = 1,
        frame: int = 0,
        gif: bool = False
    ) -> Union[Figure, FuncAnimation]:
        fac = max(1, factor)
        vector = self.vector[:, ::fac, ::fac,:]  
        locations = self.locations[::fac, ::fac, :]  
        return plot_vector_field(locations, vector, self.map, frame, gif)