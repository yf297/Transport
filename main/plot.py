import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import numpy as np
import cartopy.feature as cfeature

from . import projections
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.feature as cfeature
from . import projections

def observations(XY, Z, extent=None, frame=0, gif=False):
    """
    Parameters
    ----------
    XY : np.ndarray
        (N, 2) array of point coordinates
    Z : np.ndarray
        For a static plot, shape (T, N) or similar (then you pick one frame).
        For a GIF, shape (total_frames, N).
    extent : tuple, optional
        (lonW, lonE, latS, latN); if provided, uses Lambert projection
    frame : int, optional
        Which frame to display if gif=False
    gif : bool, optional
        If True, create and return an animation as a GIF. If False, return a static figure.
    total_frames : int, optional
        Number of frames for the GIF
    out_gif : str, optional
        Filename to save the GIF
    """

    # GIF mode
    vmin = Z.min()
    vmax = Z.max()
    if gif:
        total_frames = Z.shape[0]
        if extent is not None:
            res = "50m"

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection=projections.Lambert)

                        # Transform the extent bounds from lat/lon to Lambert x/y
            r = projections.Lambert.transform_point(extent[1], extent[3], projections.Geodetic)[0]
            l = projections.Lambert.transform_point(extent[0], extent[3], projections.Geodetic)[0]
            u = projections.Lambert.transform_point(extent[1], extent[3], projections.Geodetic)[1]
            d = projections.Lambert.transform_point(extent[0], extent[2], projections.Geodetic)[1]
            
            ax.set_extent([l, r, d, u], crs=projections.Lambert)
            ax.stock_img()
            ax.add_feature(cfeature.COASTLINE.with_scale(res))
            ax.add_feature(cfeature.STATES.with_scale(res))
            ax.gridlines(draw_labels=True)

            scatter = ax.scatter(XY[:, 0], XY[:, 1], c=Z[0], cmap="viridis", s=5,vmin=vmin, vmax=vmax)

            def update(frame_i):
                scatter.set_array(Z[frame_i])
                return (scatter,)

        else:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_aspect("equal")

            scatter = ax.scatter(XY[:, 0], XY[:, 1], c=Z[0], cmap="viridis", s=5,vmin=vmin, vmax=vmax)

            def update(frame_i):
                scatter.set_array(Z[frame_i])
                return (scatter,)

        anim = animation.FuncAnimation(fig, update, frames=total_frames, interval=200, blit=True)
        plt.close(fig)
        return anim

    # Static figure mode
    else:
        if extent is not None:
            lonW, lonE, latS, latN = extent
            res = '50m'

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection=projections.Lambert)

            r = projections.Lambert.transform_point(extent[1], extent[3], projections.Geodetic)[0]
            l = projections.Lambert.transform_point(extent[0], extent[3], projections.Geodetic)[0]
            u = projections.Lambert.transform_point(extent[1], extent[3], projections.Geodetic)[1]
            d = projections.Lambert.transform_point(extent[0], extent[2], projections.Geodetic)[1]
            

            ax.set_extent([l, r, d, u], crs=projections.Lambert)
            ax.stock_img()
            ax.add_feature(cfeature.COASTLINE.with_scale(res))
            ax.add_feature(cfeature.STATES.with_scale(res))
            ax.gridlines(draw_labels=True)

            ax.scatter(XY[:, 0], XY[:, 1], c=Z[frame], cmap="viridis", s=5,vmin=vmin, vmax=vmax)
            fig.tight_layout()
            return fig

        else:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_aspect("equal")
            ax.scatter(XY[:, 0], XY[:, 1], c=Z[frame], cmap="viridis", s=5,vmin=vmin, vmax=vmax)
            fig.tight_layout()
            return fig




def velocities(XY_UV, extent, scale = 1, frame=0, color="red", gif=False):
    """
    Creates either a static quiver plot or an animated GIF of velocity fields.

    Parameters
    ----------
    XY_UV : np.ndarray
        Array with shape (T, N, 4) where each frame contains columns [X, Y, U, V].
    extent : tuple or None
        (lonW, lonE, latS, latN); if provided, uses Lambert projection.
    frame : int
        Frame index to display in static mode.
    color : str
        Color of the quiver arrows.
    gif : bool
        If True, create and return an animated GIF.
    total_frames : int
        Total number of frames for the animation.
    out_gif : str
        Output GIF filename.

    Returns
    -------
    animation.FuncAnimation or matplotlib.figure.Figure
        The animation object (if gif=True; also saved to out_gif) or the static figure.
    """
    if gif:
        total_frames = len(XY_UV)
        if extent is not None:
            lonW, lonE, latS, latN = extent
            res = "50m"

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection=projections.Lambert)

            r = projections.Lambert.transform_point(extent[1], extent[3], projections.Geodetic)[0]
            l = projections.Lambert.transform_point(extent[0], extent[3], projections.Geodetic)[0]
            u = projections.Lambert.transform_point(extent[1], extent[3], projections.Geodetic)[1]
            d = projections.Lambert.transform_point(extent[0], extent[2], projections.Geodetic)[1]
            

            ax.set_extent([l, r, d, u], crs=projections.Lambert)
            ax.stock_img()
            ax.add_feature(cfeature.COASTLINE.with_scale(res))
            ax.add_feature(cfeature.STATES.with_scale(res))
            ax.gridlines(draw_labels=True)

            # Initialize the quiver object with the first frame.
            X0, Y0, U0, V0 = XY_UV[0].T
            q = ax.quiver(X0, Y0, U0, V0, angles='xy', scale_units='xy', scale = scale, color=color)

            def update(frame_i):
                X, Y, U, V = XY_UV[frame_i].T
                # Update the positions and vector components
                q.set_offsets(np.column_stack((X, Y)))
                q.set_UVC(U, V)
                return [q]

        else:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_aspect("equal")

            X0, Y0, U0, V0 = XY_UV[0].T
            q = ax.quiver(X0, Y0, U0, V0, angles='xy', scale_units='xy', scale = scale, color=color)

            def update(frame_i):
                X, Y, U, V = XY_UV[frame_i].T
                q.set_offsets(np.column_stack((X, Y)))
                q.set_UVC(U, V)
                return [q]
            
        anim = animation.FuncAnimation(fig, update, frames=total_frames, interval=200, blit=False)
        plt.close(fig)
        return anim

    else:
        if extent is not None:
            res = "50m"

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection=projections.Lambert)
            r = projections.Lambert.transform_point(extent[1], extent[3], projections.Geodetic)[0]
            l = projections.Lambert.transform_point(extent[0], extent[3], projections.Geodetic)[0]
            u = projections.Lambert.transform_point(extent[1], extent[3], projections.Geodetic)[1]
            d = projections.Lambert.transform_point(extent[0], extent[2], projections.Geodetic)[1]

            ax.set_extent([l, r, d, u], crs=projections.Lambert)
            ax.stock_img()
            ax.add_feature(cfeature.COASTLINE.with_scale(res))
            ax.add_feature(cfeature.STATES.with_scale(res))
            ax.gridlines(draw_labels=True)

            X0, Y0, U0, V0 = XY_UV[frame].T
            ax.quiver(X0, Y0, U0, V0, angles='xy', scale_units='xy', scale = scale, color=color)
            fig.tight_layout()
            return fig
        else:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_aspect("equal")
            X0, Y0, U0, V0 = XY_UV[frame].T
            ax.quiver(X0, Y0, U0, V0, angles='xy', scale_units='xy', scale = scale, color=color)
            fig.tight_layout()
            return fig