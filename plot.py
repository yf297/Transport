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

    contour = ax.pcolormesh(
        data.x,
        data.y,
        data.obs[0, :, :],
        cmap="coolwarm")

    cbar = plt.colorbar(contour, ax=ax, orientation='vertical', pad=0.02, aspect=16, shrink=0.8)
    cbar.set_label('Dew Point Temperature (°C)', fontsize=12)

    def update_contour(frame):
        nonlocal contour 
        contour.remove()
        contour = ax.pcolormesh(
            data.x,
            data.y,
            data.obs[frame, :, :],
            cmap="coolwarm"
            )
        
        ax.set_title(f"Dew Point Temperature at Hour {frame}", fontsize=14)
        return contour

    ani = animation.FuncAnimation(
        fig,
        update_contour,
        frames=data.time.shape[0],  
        interval=100,         
        blit=False
    )

    plt.close(fig)

    return HTML(ani.to_jshtml())


def winds(data, U, V, proj):

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
    ax.stock_img()
    ax.gridlines(
    draw_labels=True)

    step = 1

    Q = ax.quiver(
        data.x[::step],
        data.y[::step],
        U[0, ::step, ::step],
        V[0, ::step, ::step],
        transform=proj)


    def update_quiver(frame):
        Q.set_UVC(U[frame, ::step, ::step], V[frame, ::step, ::step])
        ax.set_title(f"Wind Velocities at Hour {frame+1}")
        return Q,

    ani = animation.FuncAnimation(
        fig,
        update_quiver,
        frames=U.shape[0]
    )

    plt.close(fig)

    return HTML(ani.to_jshtml())