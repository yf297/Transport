class DiscreteCoordField:
    def __init__(
        self,
        T,
        XY,
        proj,
        extent=None,
        grid=None,
    ):
        self.T = T
        self.XY = XY
        self.proj = proj
        self.extent = extent
        self.grid = grid