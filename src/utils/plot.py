import torch
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from matplotlib.animation import FuncAnimation
from typing import Optional, Union

def plot_scalar_field(
    scalar: torch.Tensor,
    map: Optional[object] = None,
    frame: int = 0,
    gif: bool = False
) -> Union[plt.Figure, FuncAnimation]:
    """
    Plot a single frame or animation of a time-indexed scalar tensor.

    Parameters
    ----------
    scalar : torch.Tensor
        Tensor of shape (N, H, W) containing scalar field values.
    map : object, optional
        An object with attributes `proj` (Cartopy CRS) and `extent` (tuple)
        for projection. If None, a plain axes with equal aspect is used.
    frame : int
        Which time index to display (ignored if `gif=True`).
    gif : bool
        If True, returns a FuncAnimation over all frames.
    """
    scalar_np = scalar.numpy() 
    vmin, vmax = scalar_np.min(), scalar_np.max()

    fig = plt.figure()
    if map is not None:
        ax = fig.add_subplot(1, 1, 1, projection=map.proj)
        ax.set_extent(map.extent, crs=map.proj)
        ax.stock_img()
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"))
        ax.add_feature(cfeature.STATES.with_scale("50m"))
        extent = map.extent
        transform = map.proj
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect("equal")
        extent = None
        transform = None

    img = ax.imshow(
        scalar_np[0],
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        transform=transform
    )

    if gif:
        def update(i):
            img.set_data(scalar_np[i])
            return (img,)

        anim = FuncAnimation(
            fig, update,
            frames=scalar_np.shape[0],
            interval=200,
            blit=True
        )
        plt.close(fig)
        return anim
    else:
        img.set_data(scalar_np[frame])
        fig.tight_layout()
        plt.close(fig)
        return fig



def plot_vector_field(
    locations: torch.Tensor,
    vectors: torch.Tensor,
    map: Optional[object] = None,
    frame: int = 0,
    gif: bool = False
) -> Union[plt.Figure, FuncAnimation]:
    """
    Plot a static or animated vector field at arbitrary locations.

    Parameters
    ----------
    locations : torch.Tensor
        Shape (H, W, 2), the (x,y) points where vectors are defined.
    vectors : torch.Tensor
        Shape (N, H, W, 2), the (u,v) vectors at each time step.
    map : object, optional
        An object with attributes `proj` (Cartopy CRS) and `extent` (tuple)
        for projection. If None, a plain axes with equal aspect is used.
    frame : int
        Which time index to display (ignored if `gif=True`).
    gif : bool
        If True, returns a FuncAnimation over all frames.
    """
    loc_np = locations.numpy()      
    vec_np = vectors.detach().numpy()  
    xs, ys = loc_np[:,:,0], loc_np[:,:,1]
    N = vec_np.shape[0]

    fig = plt.figure()
    if map:
        ax = fig.add_subplot(1,1,1, projection=map.proj)
        ax.set_extent(map.extent, crs=map.proj)
        ax.stock_img()
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"))
        ax.add_feature(cfeature.STATES.with_scale("50m"))
        transform = map.proj
    else:
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect("equal")
        transform = None

    if gif:
        Q = ax.quiver(
            xs, ys,
            vec_np[0,:,:,0], vec_np[0,:,:,1],
            angles="xy", scale_units="xy",
            transform=transform
        )
        def _update(i):
            Q.set_UVC(vec_np[i,:,:,0], vec_np[i,:,:,1])
            return (Q,)

        anim = FuncAnimation(
            fig, _update, frames=N,
            interval=200, blit=False
        )
        plt.close(fig)
        return anim

    ax.quiver(
        xs, ys,
        vec_np[frame,:,:,0], vec_np[frame,:,:,1],
        angles="xy", scale_units="xy",
        transform=transform
    )
    fig.tight_layout()
    plt.close(fig)
    return fig
   

