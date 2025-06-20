import matplotlib.pyplot as plt
import matplotlib.animation
import torch
import numpy as np
import cartopy.feature
from scipy.interpolate import griddata

def scalar_field(TXY, Z, proj=None, extent=None):
    Z = Z.detach()
    vmin, vmax = Z.min(), Z.max()
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent(extent, crs=proj)
    ax.stock_img()
    ax.add_feature(cartopy.feature.COASTLINE.with_scale("50m"))
    ax.add_feature(cartopy.feature.STATES.with_scale("50m"))

    idx = [torch.nonzero(TXY[:,0] == t_i, as_tuple=False).squeeze(1) for t_i in torch.unique(TXY[:,0])]
    grid_res = 100
    x_min, x_max = TXY[:,1].min().item(), TXY[:,1].max().item()
    y_min, y_max = TXY[:,2].min().item(), TXY[:,2].max().item()
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res)
    )

    # Interpolate first frame
    x, y = TXY[idx[0], 1].numpy(), TXY[idx[0], 2].numpy()
    z = Z[idx[0]].numpy()
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
    img = ax.imshow(
        grid_z, origin='lower',
        extent=[x_min, x_max, y_min, y_max],
        cmap='magma', vmin=vmin, vmax=vmax, interpolation='bicubic'
    )

    def update(i):
        x = TXY[idx[i], 1].numpy()
        y = TXY[idx[i], 2].numpy()
        z = Z[idx[i]].numpy()
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
        img.set_data(grid_z)
        return (img,)

    anim = matplotlib.animation.FuncAnimation(
        fig, update,
        frames=len(idx),
        interval=200,
        blit=False
    )
    plt.close(fig)
    return anim



def vector_field(TXY, UV, proj=None, extent=None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent(extent, crs=proj)
    ax.stock_img()
    ax.add_feature(cartopy.feature.COASTLINE.with_scale("50m"))
    ax.add_feature(cartopy.feature.STATES.with_scale("50m"))

    idx = [torch.nonzero(TXY[:,0] == t_i, as_tuple=False).squeeze(1) for t_i in torch.unique(TXY[:,0])]
    x, y = TXY[idx[0], 1].numpy(), TXY[idx[0], 2].numpy()
    u = UV[idx[0], 0].detach().numpy()
    v = UV[idx[0], 1].detach().numpy()
   
    Q = ax.quiver(x=x, y=y, u=u, v=v, scale = 300, pivot="mid" )

    def _update(i):
        x = TXY[idx[i], 1].numpy()
        y = TXY[idx[i], 2].numpy()
        u = UV[idx[i], 0].detach().numpy()
        v = UV[idx[i], 1].detach().numpy()
        Q.set_offsets(np.c_[x, y])
        Q.set_UVC(u, v)
        return (Q,)

    anim = matplotlib.animation.FuncAnimation(
        fig, _update,
        frames=len(idx),
        interval=200,
        blit=False
    )
    plt.close(fig)
    return anim
 
