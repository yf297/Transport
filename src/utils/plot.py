import torch
import numpy as np
import matplotlib.figure
import matplotlib.animation
import matplotlib.pyplot
import cartopy.feature
from typing import Union, Tuple, Optional


def scalar_field(
    Z: torch.Tensor,
    proj: Optional[object] = None,
    extent: Optional[Tuple[float, float, float, float]] = None,
    frame: int = 0,
    gif: bool = False
) -> Union[matplotlib.figure.Figure, matplotlib.animation.FuncAnimation]:
    
    Z_np = Z.numpy() 
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

    img = ax.imshow(
        Z_np[0],
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        transform=transform
    )

    if gif:
        def update(i):
            img.set_data(Z_np[i])
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
        img.set_data(Z_np[frame])
        fig.tight_layout()
        matplotlib.pyplot.close(fig)
        return fig


def vector_field(
    XY: torch.Tensor,
    UV: torch.Tensor,
    proj: Optional[object] = None,
    extent: Optional[Tuple[float, float, float, float]] = None,
    frame: int = 0,
    gif: bool = False
) -> Union[matplotlib.figure.Figure, matplotlib.animation.FuncAnimation]:

    XY_np = XY.numpy()
    UV_np = UV.detach().numpy()
    xs, ys = XY_np[:, 0], XY_np[:, 1]
    N = UV_np.shape[0]

    #x_range = xs.max() - xs.min()
    #y_range = ys.max() - ys.min()
    #spacing = np.sqrt((x_range * y_range) / N)
    scale = 3e-4

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
            width=0.003,             # thinner shafts
            headwidth=4,             # wider heads
            headlength=6,            # longer heads
            headaxislength=5,        # stem length inside head
            pivot="mid",             # arrows centered on the point
            cmap="viridis",      
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
            width=0.003,             # thinner shafts
            headwidth=3,             # wider heads
            headlength=4,            # longer heads
            headaxislength=5,        # stem length inside head
            pivot="mid",             # arrows centered on the point
            color="red",      
            transform=transform
        )
        fig.tight_layout()
        matplotlib.pyplot.close(fig)
        return fig
