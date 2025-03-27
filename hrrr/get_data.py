from herbie import Herbie
import xarray as xr 
from datetime import date, datetime, timedelta
import torch 
import sys
import os

# Allow imports from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import tools, projections, model

def process_hrrr(path, i):
    """
    Open a single HRRR GRIB file as an xarray Dataset and expand its 
    dimensions to include a 'time' coordinate.

    Parameters:
    -----------
    path : str
        The file path to the HRRR GRIB file.
    i : int
        Time index (e.g., forecast hour index).

    Returns:
    --------
    ds : xarray.Dataset
        The opened and updated dataset with an added 'time' dimension.
    """

    ds = xr.open_dataset(path, engine='cfgrib')
    ds.expand_dims(dim={"time": [i]})
    return ds

def hrrr(date, level, hours, extent, factor):
    """
    Download and process HRRR weather model data for a specific date, 
    returning time, spatial coordinates, temperature (°C), and combined (X, Y, U, V) tensors.

    Parameters:
    -----------
    date : str
        The date for which the HRRR data is needed (e.g., '2023-01-01').
    level : str
        The atmospheric level or parameter to download (e.g., '500 mb', '700 mb').
    hours : int
        The number of forecast hours from the start of the day (0 to 23).
    extent : tuple
        A bounding box (min_lon, max_lon, min_lat, max_lat) for spatial subsetting.
    factor : int
        Downsampling factor for the dataset (e.g., keep every nth grid point).

    Returns:
    --------
    T : torch.Tensor
        1D tensor of time points (in days, from 0 to hours/24), size = [total_times].
    XY : torch.Tensor
        2D tensor of spatial coordinates, size = [num_points, 2].
        Each row is [x_coord, y_coord] transformed into Lambert projection.
    Z : torch.Tensor
        2D tensor of temperatures (°C), size = [total_times, num_points].
    XY_UV : list of torch.Tensor
        A list (length = total_times) of tensors, each size = [num_points, 4].
        Each tensor row is [x_coord, y_coord, u_wind, v_wind].
    """

    # Prepare the list of file paths for each forecast hour
    paths = []


    # Download HRRR files for each forecast hour
    for i in range(hours):
        paths.append(Herbie(date, model="hrrr", fxx=i).download(level))

    # Process all downloaded files and concatenate along the 'time' dimension
    data = xr.concat(
        [process_hrrr(paths[i], i) for i in range(hours)], 
        dim='time'
    )

    # Adjust longitude values from [0..360] to [-180..180]
    data["longitude"] = data["longitude"] - 360

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
    l =  projections.Lambert.transform_point(extent[0], extent[3], projections.Geodetic)[0]
    u = projections.Lambert.transform_point(extent[1], extent[3], projections.Geodetic)[1]
    d = projections.Lambert.transform_point(extent[0], extent[2], projections.Geodetic)[1]
    
    # Select the region of interest based on the transformed extent
    data_full = data.sel(x=slice(l, r), y=slice(d, u))
    data = data_full.isel(y=slice(None, None, factor), x=slice(None, None, factor))
    
    # Extract temperature (in K), convert to °C, and reshape
    Z_full = torch.tensor(data_full.dpt.values, dtype=torch.float32).reshape(hours, -1) - 273.15
    Z = torch.tensor(data.dpt.values, dtype=torch.float32).reshape(hours, -1) - 273.15

    # Get the final XY coordinates
    XY_full = torch.tensor(
        projections.Lambert.transform_points(
        projections.Geodetic, 
        data_full.longitude.values,
        data_full.latitude.values
        )[:, :, 0:2],
        dtype=torch.float32
    ).reshape(-1, 2)
    
    XY = torch.tensor(
        projections.Lambert.transform_points(
        projections.Geodetic, 
        data.longitude.values,
        data.latitude.values
        )[:, :, 0:2],
        dtype=torch.float32
    ).reshape(-1, 2)

    # Create a 1D time tensor from 0 to hours/24
    T = torch.linspace(0, hours / 24, hours)
    
    level_num = float(level.split()[0])  # Takes the first part (e.g., "700" from "700 mb")
    P = torch.full((XY.shape[0],1), level_num, dtype=torch.float32)

    # Create a list of tensors, each containing [x, y, p, u, v] for each time step
    XYP_UV = [
        torch.cat([
            XY, P,
            torch.tensor(data.u.values.reshape(hours, -1, 1)[i]),
            torch.tensor(data.v.values.reshape(hours, -1, 1)[i])
        ], dim=-1)
        for i in range(len(T))
    ]
    data = model.data(T, XY, Z, XYP_UV, XY_full, Z_full)
    data.extent = extent
    data.date = date
    data.level = level
    return data