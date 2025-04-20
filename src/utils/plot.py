import torch
import matplotlib.figure
import matplotlib.animation
import matplotlib.pyplot
import cartopy.feature
from typing import Union, Tuple, Optional

def scalar_field(
    scalar: torch.Tensor,
    proj: Optional[object] = None,
    extent: Optional[Tuple[float, float, float, float]] = None,
    frame: int = 0,
    gif: bool = False
) -> Union[matplotlib.figure.Figure, matplotlib.animation.FuncAnimation]:
    scalar_np = scalar.numpy() 
    vmin, vmax = scalar_np.min(), scalar_np.max()
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

        anim = matplotlib.animation.FuncAnimation(
            fig, update,
            frames=scalar_np.shape[0],
            interval=200,
            blit=True
        )
        matplotlib.pyplot.close(fig)
        return anim
    else:
        img.set_data(scalar_np[frame])
        fig.tight_layout()
        matplotlib.pyplot.close(fig)
        return fig



def vector_field(
    locations: torch.Tensor,
    vector: torch.Tensor,
    proj: Optional[object] = None,
    extent: Optional[Tuple[float, float, float, float]] = None,
    frame: int = 0,
    gif: bool = False
) -> Union[matplotlib.figure.Figure, matplotlib.animation.FuncAnimation]:
    loc_np = locations.numpy()      
    vec_np = vector.detach().numpy()  
    xs, ys = loc_np[:,:,0], loc_np[:,:,1]
    N = vec_np.shape[0]
    fig =  matplotlib.pyplot.figure()
    
    if proj:
        ax = fig.add_subplot(1,1,1, projection=proj)
        ax.set_extent(extent, crs=proj)
        ax.stock_img()
        ax.add_feature(cartopy.feature.COASTLINE.with_scale("50m"))
        ax.add_feature(cartopy.feature.STATES.with_scale("50m"))
        transform = proj
    else:
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect("equal")
        transform = None

    if gif:
        Q = ax.quiver(
            xs, ys,
            vec_np[0,:,:,0], vec_np[0,:,:,1],
            angles="xy", scale_units="xy", color = "red",
            transform=transform
        )
        def _update(i):
            Q.set_UVC(vec_np[i,:,:,0], vec_np[i,:,:,1])
            return (Q,)

        anim = matplotlib.animation.FuncAnimation(
            fig, _update, frames=N,
            interval=200, blit=False
        )
        matplotlib.pyplot.close(fig)
        return anim

    ax.quiver(
        xs, ys,
        vec_np[frame,:,:,0], vec_np[frame,:,:,1],
        angles="xy", scale_units="xy", color = "red",
        transform=transform
    )
    fig.tight_layout()
    matplotlib.pyplot.close(fig)
    return fig
   

