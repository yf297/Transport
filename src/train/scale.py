class ScaleT:
    def __init__(self, T):
        self.min   = T.min()
        self.range = (T.max() - self.min)      # ← use max–min
    def __call__(self, t):
        return (t.to(t.device) - self.min.to(t.device)) \
               / self.range.to(t.device)
               
class ScaleXY:
    def __init__(self, XY):
        self.center = XY.mean(dim=0)                 
        self.scale  = XY.std(dim=0, unbiased=False)

    def __call__(self, xy):
        return (xy - self.center.to(xy.device)) / self.scale.to(xy.device)

class DeScaleA:
    def __init__(self, scaleXY):
        self.center = scaleXY.center
        self.scale  = scaleXY.scale

    def __call__(self, a):
        return a * self.scale.to(a.device) + self.center.to(a.device)

class ScaleZ:
    def __init__(
        self,
        Z
    ):
        flat = Z.reshape(-1)
        self.mean = flat.mean()
        self.scale = flat.std(unbiased=False)

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
