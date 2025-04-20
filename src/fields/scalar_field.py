import torch
import matplotlib.figure
import matplotlib.animation
import utils.plot
import fields.coord_field
from typing import Union

class DiscreteScalarField:
    def __init__(
        self,
        coord_field: fields.coord_field.DiscreteCoordField,
        scalar: torch.Tensor,
    ):
        self.coord_field = coord_field
        self.scalar = scalar

    def plot(
        self,
        factor: int = 1,
        frame: int = 0,
        gif: bool = False
    ) -> Union[matplotlib.figure.Figure, matplotlib.animation.FuncAnimation]:
        fac = max(1, factor)
        scalar = self.scalar[:, ::fac, ::fac]
        
        return utils.plot.scalar_field(
            scalar=scalar,
            proj=self.coord_field.proj,
            extent=self.coord_field.extent,
            frame=frame,
            gif=gif,
        )