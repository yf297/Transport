class ScaleT:
    def __init__(
        self, 
        T
    ):
        self.scale = T.max()
        
    def __call__(
        self,
        t
    ):
        return t / self.scale.to(t.device)

class ScaleXY:
    def __init__(
        self,
        XY
    ):  
        xy_min = XY.min(dim=0).values        
        xy_max = XY.max(dim=0).values        
        self.center = (xy_min + xy_max) / 2  
        half_ranges = (xy_max - xy_min) / 2  
        self.scale = half_ranges.max()

    def __call__(
        self, 
        xy
    ):
        return (xy - self.center.to(xy.device)) / self.scale.to(xy.device)

class DeScaleA:
    def __init__(
        self, 
        scaleXY
    ):
        self.center = scaleXY.center
        self.scale = scaleXY.scale

    def __call__(
        self, 
        a
    ):
        return a * self.scale.to(a.device) + self.center.to(a.device)

class ScaleZ:
    def __init__(
        self,
        Z
    ):
        flat = Z[0, :]
        self.mean = flat.mean()
        self.scale = flat.std()

    def __call__(
        self,
        z
    ):
        return (z -  self.mean.to(z.device)) / self.scale.to(z.device)

class DeScaleFlow:
    def __init__(
        self,
        scaled_flow,
        scale_T,
        scale_XY,
        descale_A
    ):
        self.scale_T = scale_T
        self.scale_XY = scale_XY
        self.descale_A = descale_A
        self.scaled_flow = scaled_flow

    def __call__(
        self,
        T, 
        XY
    ):
        scaled_t = self.scale_T(T)
        scaled_XY = self.scale_XY(XY)
        scaled_A = self.scaled_flow(scaled_t, scaled_XY)
        A = self.descale_A(scaled_A)
        return A
