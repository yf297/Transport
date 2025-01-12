import cartopy.feature as cfeature
import cartopy.crs as ccrs

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

def obs(data, proj):

    latN = 50.4
    latS = 23
    lonW = -123
    lonE = -73
    res = '50m'

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    ax.set_extent([lonW, lonE, latS, latN], crs=ccrs.Geodetic())

    ax.add_feature(cfeature.COASTLINE.with_scale(res))
    ax.add_feature(cfeature.STATES.with_scale(res))


    contour = ax.contourf(
        data.X,
        data.Y,
        data.obs[0, :, :],
        cmap="coolwarm")

    cbar = plt.colorbar(contour, ax=ax, orientation='vertical', pad=0.02, aspect=16, shrink=0.8)
    cbar.set_label('Dew Point Temperature (°C)', fontsize=12)

    def update_contour(frame):
        nonlocal contour 
        contour.remove()
        contour = ax.contourf(
            data.X,
            data.Y,
            data.obs[frame, :, :],
            cmap="coolwarm",
            levels=20
        )
        ax.set_title(f"Dew Point Temperature at Hour {frame}", fontsize=14)
        return contour

    ani = animation.FuncAnimation(
        fig,
        update_contour,
        frames=data.time.shape[0],  
        interval=200,         
        blit=False
    )

    plt.close(fig)

    return HTML(ani.to_jshtml())