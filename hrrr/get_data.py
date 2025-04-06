from herbie import Herbie
import xarray as xr 
from datetime import date, datetime, timedelta
import torch 
import numpy as np
import sys
import os

# Allow imports from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import tools, projections, model

def process_hrrr(path, i):

    ds = xr.open_dataset(path, engine='cfgrib')
    ds.expand_dims(dim={"time": [i]})
    
    return ds

def hrrr(date, level, hours, extent):

    paths = []

    # Download HRRR files for each forecast hour
    for i in range(0, hours + 1):
        paths.append(Herbie(date, model="hrrr", fxx=i).download(level))

    # Process all downloaded files and concatenate along the 'time' dimension
    data = xr.concat(
        [process_hrrr(paths[i], i) for i in range(0, hours + 1)], 
        dim='time'
    )

    # Adjust longitude values from [0..360] to [-180..180]
    data["longitude"] = data["longitude"] - 360
    # Flip latitude values
    data = data.isel(y=slice(None, None, -1))

    # Transform latitude & longitude into x & y using Lambert projection
    grid = projections.Lambert.transform_points(
        projections.Geodetic, 
        data.longitude.values,
        data.latitude.values
    )[:, :, 0:2]

    # Assign x and y coordinates to the dataset
    data = data.assign_coords(x=("x", grid[0, :, 0]))
    data = data.assign_coords(y=("y", grid[:, 0, 1]))

    # Transform the extent bounds from lat/lon to Lambert x/y
    r = projections.Lambert.transform_point(extent[1], extent[3], projections.Geodetic)[0]
    l = projections.Lambert.transform_point(extent[0], extent[3], projections.Geodetic)[0]
    u = projections.Lambert.transform_point(extent[1], extent[3], projections.Geodetic)[1]
    d = projections.Lambert.transform_point(extent[0], extent[2], projections.Geodetic)[1]
    
    # Select the region of interest based on the transformed extent
    data = data.sel(x=slice(l, r), y=slice(u, d))
   # data = data.isel(x=slice(None, None, 3), y=slice(None, None, 3))
    Z = torch.tensor(np.array(data.dpt.values), dtype=torch.float32).reshape(hours + 1, -1) 

    # Get the final XY coordinates
    XY_grid = torch.tensor(
        projections.Lambert.transform_points(
        projections.Geodetic, 
        data.longitude.values,
        data.latitude.values
        )[:, :, 0:2],
        dtype=torch.float32
    )
    XY = XY_grid.reshape(-1, 2)
    
    T = torch.linspace(0, hours * 3600, hours + 1)

    # Create a list of tensors, each containing [x, y, p, u, v] for each time step
    XY_UV = [
        torch.cat([
            XY,
            torch.tensor(data.u.values.reshape(hours + 1, -1, 1)[i]),
            torch.tensor(data.v.values.reshape(hours + 1, -1, 1)[i])
        ], dim=-1)
        for i in range(len(T))
    ]
    data = model.data(T, XY, Z, XY_UV, grid_size = 1, k0 = 0, k1 = 1)
    data.extent = extent
    data.date = date
    data.level = level
    return data