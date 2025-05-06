import torch
import matplotlib.animation
import matplotlib.pyplot
import cartopy.feature

def scalar_field(
    XY,
    Z,
    proj=None,
    extent=None,
    frame=0,
    gif=False
):
    
    Z_np = Z.numpy() 
    XY_np = XY.numpy()
    xs, ys = XY_np[:, 0], XY_np[:, 1]
    vmin, vmax = Z_np.min(), Z_np.max()
    fig = matplotlib.pyplot.figure()
    
    if proj is not None:
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_extent(extent, crs=proj)
        ax.stock_img()
        ax.add_feature(cartopy.feature.COASTLINE.with_scale("50m"))
        ax.add_feature(cartopy.feature.STATES.with_scale("50m"))
        extent = extent
        transform = proj
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect("equal")
        extent = None
        transform = None

    img = ax.scatter(xs, ys, c=Z[0], cmap="viridis", s=5, vmin=vmin, vmax=vmax)


    if gif:
        def update(i):
            img.set_array(Z[i])
            return (img,)

        anim = matplotlib.animation.FuncAnimation(
            fig, update,
            frames=Z_np.shape[0],
            interval=200,
            blit=True
        )
        matplotlib.pyplot.close(fig)
        return anim
    else:
        img.set_array(Z_np[frame])
        fig.tight_layout()
        matplotlib.pyplot.close(fig)
        return fig


def vector_field(
    XY,
    UV,
    proj=None,
    extent=None,
    frame=0,
    gif=False
):

    XY_np = XY.numpy()
    UV_np = UV.detach().numpy()
    xs, ys = XY_np[:, 0], XY_np[:, 1]
    N = UV_np.shape[0]

    scale = 2e-4

    fig = matplotlib.pyplot.figure()
    if proj:
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_extent(extent, crs=proj)
        ax.stock_img()
        ax.add_feature(cartopy.feature.COASTLINE.with_scale("50m"))
        ax.add_feature(cartopy.feature.STATES.with_scale("50m"))
        transform = proj
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect("equal")
        transform = None

    if gif:
        Q = ax.quiver(
            xs, ys,
            UV_np[0, :, 0], UV_np[0, :, 1],
            angles="xy", scale_units="xy",
            scale=scale, 
            width=0.003,             
            headwidth=4,             
            headlength=6,            
            headaxislength=5,       
            pivot="mid", 
            color="red",      
            transform=transform
        )
        def _update(i):
            Q.set_UVC(UV_np[i, :, 0], UV_np[i, :, 1])
            return (Q,)

        anim = matplotlib.animation.FuncAnimation(
            fig, _update,
            frames=N,
            interval=200,
            blit=False
        )
        matplotlib.pyplot.close(fig)
        return anim
    else:
        ax.quiver(
            xs, ys,
            UV_np[frame, :, 0], UV_np[frame, :, 1],
            angles="xy", scale_units="xy",
            scale=scale, 
            width=0.003,           
            headwidth=3,          
            headlength=4,           
            headaxislength=5,       
            pivot="mid",             
            color="red",      
            transform=transform
        )
        fig.tight_layout()
        matplotlib.pyplot.close(fig)
        return fig
