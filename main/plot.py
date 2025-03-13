import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import numpy as np
import cartopy.feature as cfeature

from . import tools
def observations(XY, Z, extent=None, frame=0):
    
    if extent is not None:
        lonW, lonE, latS, latN = extent
        res = '50m'

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=tools.Lambert_proj)
        ax.set_extent([lonW, lonE, latS, latN], crs=tools.Geodetic_proj)
        ax.add_feature(cfeature.COASTLINE.with_scale(res))
        ax.add_feature(cfeature.STATES.with_scale(res))
        ax.stock_img()
        ax.gridlines()
            
        ax.scatter(XY[:, 0], XY[:, 1], c=Z[frame], cmap="viridis", s=10)
        fig.tight_layout()
        return fig

    else:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect("equal")
        ax.scatter(XY[:, 0], XY[:, 1], c=Z[frame], cmap="viridis", s=10)
        fig.tight_layout()
        return fig


def velocities(XY_UV, extent, frame=0, color="red", ax=None):
    import matplotlib.pyplot as plt
    import cartopy.feature as cfeature
    import cartopy.crs as ccrs
    # from your code, e.g. tools.Lambert_proj, tools.Geodetic_proj

    if ax is None:
        # Create a new figure+axes
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=tools.Lambert_proj)
    else:
        # If an Axes was passed in, get its Figure
        fig = ax.figure

    # If you have an extent, set up map features
    if extent is not None:
        lonW, lonE, latS, latN = extent
        ax.set_extent([lonW, lonE, latS, latN], crs=tools.Geodetic_proj)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"))
        ax.add_feature(cfeature.STATES.with_scale("50m"))
        ax.gridlines(draw_labels=True)
        ax.stock_img()

    # Plot quiver
    X0, Y0, U0, V0 = XY_UV[frame].T
    ax.quiver(X0, Y0, U0, V0,
              #angles='xy',
              #scale_units='xy', 
              #scale=10,
              color=color)

    fig.tight_layout()
    return fig, ax
