# src/fields/scalar_field.py
import utils.plot

class DiscreteScalarField:
    def __init__(
        self,
        coord_field,
        Z
    ):
        self.coord_field = coord_field
        self.Z = Z
        
    def plot(
        self,
        factor=1,
        frame=0,
        gif=False
    ):
        Z = self.Z
        XY = self.coord_field.XY
        n = Z.size(0)
        if self.coord_field.grid is not None:
            k1 = self.coord_field.grid[0]
            k2 = self.coord_field.grid[1]
            fac = max(1, factor)

            XY = XY.reshape(k1, k2, 2)
            XY = XY[::fac, ::fac, :].reshape(-1,2)
            
            Z = self.Z.reshape(n, k1, k2)
            Z = Z[:, ::fac, ::fac].reshape(n, -1)
        
        return utils.plot.scalar_field(
            XY=XY,
            Z=Z,
            proj=self.coord_field.proj,
            extent=self.coord_field.extent,
            frame=frame,
            gif=gif
        )
