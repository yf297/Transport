class Scale:
    def __init__(self, T, XY):
        self.T_min = T.min()
        self.T_max = T.max()
        self.T_scale = self.T_max - self.T_min
        
        self.XY_center = XY.mean(dim=0)
        self.XY_scale  = XY.std(dim=0, unbiased=False)

    def t(self, t):
        return (t - self.T_min) / self.T_scale.to(t.device)

    def xy(self, xy):
        return (xy - self.XY_center.to(xy.device)) / self.XY_scale.to(xy.device)
    
    def a(self, a):
        return a * self.XY_scale.to(a.device) + self.XY_center.to(a.device)


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

