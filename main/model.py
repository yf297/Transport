import torch
import gpytorch
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from . import plot, ode, tools

class GP(gpytorch.models.ExactGP):
    def __init__(self, kernel, likelihood):
        super(GP, self).__init__(None, None, likelihood)
        self.mean = gpytorch.means.ConstantMean()
        self.kernel = kernel
        
    def forward(self, points):
        mean = self.mean(points)
        covar = self.kernel(points, points)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class data():
    def __init__(self, T, XY, Z, XY_UV = None):
        self.T = T
        self.XY = XY
        self.Z = Z
        self.XY_UV = XY_UV
        
        self.extent = None
        self.date = None
        self.level = None
        self.minutes = None
        self.time = None
        
        self.indices = None
        self.gp = None
        self.flow = None
        
        self.input_std = torch.ones(1)
        self.output_std = torch.ones(1)
        
        self.n = self.T.shape[0]
        self.m = self.XY.shape[0]
        
    def plot_observations(self, indices, frame = 0):
        XY_ = self.XY
        Z_ = self.Z
     
        if indices is not None:
            XY_ = XY_[indices,:]
            Z_ =  [Z[indices] for Z in Z_]
            
        return plot.observations(XY_, Z_, self.extent, frame)
    
    
    def plot_vel_data(self, indices = None, frame = 0, color = "blue", ax = None):
        T_ = self.T
        XY_UV_ = self.XY_UV
        if indices is not None:
            XY_UV =  [XY_UV_[i][indices,:] for i in range(0,T_.shape[0])]
        else:
            XY_UV = XY_UV_
        return plot.velocities(XY_UV, self.extent, frame, color, ax = ax)
    
    
    def plot_vel(self, indices, frame = 0, color = "red", ax = None):
        T_ = self.T
        XY_ = self.XY
        
        XY_ = XY_[indices, ]
        UV = torch.ones((T_.shape[0], XY_.shape[0], 2))
        
        vel = ode.Vel_hat(self)
        for frame in range(0,T_.shape[0]):
            UV[frame,:,:] = vel(T_[frame], XY_)
        
        XY_UV = [torch.cat([XY_,
                            UV[i,:,0:1],
                            UV[i,:,1:2]],
                dim = -1).detach() for i in range(0,T_.shape[0])]
        return plot.velocities(XY_UV, self.extent, frame, color, ax=ax)
       
       

    def plot_both(self, indices, frame=0):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=tools.Lambert_proj)
        
        # 2) Set map extent and background
        lonW, lonE, latS, latN = self.extent
        ax.set_extent([lonW, lonE, latS, latN], crs=tools.Geodetic_proj)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"))
        ax.add_feature(cfeature.STATES.with_scale("50m"))
        ax.stock_img()
        ax.gridlines(draw_labels=True)

        XY_UV_data = self.XY_UV
        Xd, Yd, Ud, Vd = XY_UV_data[frame].T  # shape [N_data]

        XY_ = self.XY[indices]
        UV = torch.ones((len(self.T), XY_.shape[0], 2))

        vel = ode.Vel_hat(self)
        for f in range(len(self.T)):
            UV[f] = vel(self.T[f], XY_) * (1.0 / 86400.0)

        XY_UV_model = [
            torch.cat([XY_, UV[f, :, 0:1], UV[f, :, 1:2]], dim=-1).detach()
            for f in range(len(self.T))
        ]
        Xm, Ym, Um, Vm = XY_UV_model[frame].T  

        X_all = torch.cat([Xd, Xm])
        Y_all = torch.cat([Yd, Ym])
        U_all = torch.cat([Ud, Um])
        V_all = torch.cat([Vd, Vm])

        colors = ["red"] * len(Xd) + ["blue"] * len(Xm)

        ax.quiver(
            X_all, Y_all, U_all, V_all,
            angles='xy', scale_units='xy',
            color=colors
        )

        fig.tight_layout()
        return fig, ax
