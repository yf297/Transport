import xarray as xr 
from goes2go.data import goes_nearesttime
import numpy as np
import torch 
import sys
import os

# Allow imports from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import tools, projections, model

def calculate_degrees_goes(data):
    """
    Convert the x, y coordinates in a GOES projected dataset to longitude and latitude in degrees.

    Parameters:
    -----------
    data : xarray.Dataset
        A GOES dataset containing 'x' and 'y' coordinates, as well as the
        'goes_imager_projection' variable with GOES projection parameters.

    Returns:
    --------
    abi_lon : numpy.ndarray
        Longitudes (in degrees) corresponding to the GOES x, y grid, reversed along one axis.
    abi_lat : numpy.ndarray
        Latitudes (in degrees) corresponding to the GOES x, y grid, reversed along one axis.

    Notes:
    ------
    This function implements standard GOES-R projection math to convert scan angles
    into longitude/latitude. The final arrays are reversed (`[::-1]`) to match the dataset's
    orientation.
    """
    # Extract coordinate arrays
    x_coordinate_1d = data['x'][:] 
    y_coordinate_1d = data['y'][:]

    # Projection parameters from GOES-R
    projection_info = data['goes_imager_projection']
    lon_origin = projection_info.longitude_of_projection_origin
    H = projection_info.perspective_point_height + projection_info.semi_major_axis
    r_eq = projection_info.semi_major_axis
    r_pol = projection_info.semi_minor_axis
    
    # Create 2D mesh of x, y
    x_coordinate_2d, y_coordinate_2d = np.meshgrid(x_coordinate_1d, y_coordinate_1d)
    
    # Convert to radians
    lambda_0 = (lon_origin * np.pi) / 180.0

    # Intermediate variables for GOES-R projection
    a_var = (
        np.sin(x_coordinate_2d)**2 
        + np.cos(x_coordinate_2d)**2 
          * (np.cos(y_coordinate_2d)**2 
             + (r_eq**2 / r_pol**2) * np.sin(y_coordinate_2d)**2)
    )
    b_var = -2.0 * H * np.cos(x_coordinate_2d) * np.cos(y_coordinate_2d)
    c_var = (H**2) - (r_eq**2)

    # Solve for the distance from satellite to point on the Earth
    r_s = (-b_var - np.sqrt(b_var**2 - 4.0 * a_var * c_var)) / (2.0 * a_var)
    
    # Satellite coordinates (s_x, s_y, s_z)
    s_x = r_s * np.cos(x_coordinate_2d) * np.cos(y_coordinate_2d)
    s_y = -r_s * np.sin(x_coordinate_2d)
    s_z = r_s * np.cos(x_coordinate_2d) * np.sin(y_coordinate_2d)
    
    np.seterr(all='ignore')  # Suppress runtime warnings for invalid operations

    # Calculate latitude, longitude in degrees
    abi_lat = (180.0 / np.pi) * np.arctan(
        (r_eq**2 / r_pol**2) 
        * (s_z / np.sqrt((H - s_x)**2 + s_y**2))
    )
    abi_lon = (lambda_0 - np.arctan(s_y / (H - s_x))) * (180.0 / np.pi)
    
    return abi_lon, abi_lat


def goes(date, hours, band, extent, factor):
    """
    Download and process GOES satellite data (multi-band imagery and derived winds)
    for a given date and time range, returning time, spatial coordinates, 
    brightness temperature (°C), and wind vectors in the chosen bounding box.

    Parameters:
    -----------
    date : str
        Date string in 'YYYY-MM-DD' format (e.g., '2023-01-01').
    hours : int
        Number of forecast hours to generate time steps (e.g., 0 to 23).
    band : str
        GOES band name to extract from the MCMIP product (e.g., "CMI_C13").
    extent : tuple
        A bounding box (min_lon, max_lon, min_lat, max_lat) for subsetting data
        in Lambert-projected coordinates.
    factor : int
        Downsampling factor for the dataset in the x and y dimensions.

    Returns:
    --------
    T : torch.Tensor
        1D tensor of fractional day times (0 to hours/24), size = [total_times].
    XY : torch.Tensor
        2D tensor of spatial coordinates [x, y] in Lambert projection, 
        size = [num_points, 2].
    Z : list of torch.Tensor
        List (length = total_times) of brightness temperatures (°C) for each time, 
        each size = [num_points]. The values are masked to the specified extent.
    XY_UV : list of torch.Tensor
        List of 2D tensors (length = total_times). Each contains columns [x, y, u, v],
        giving wind vectors (u, v) at Lambert-projected coordinates (x, y) 
        within the bounding box.

    Notes:
    ------
    - The function uses `goes2go.data.goes_nearesttime` to fetch GOES data at times 
      generated in 30-minute steps from 00:00 up to `hours`.
    - The brightness temperature is converted to °C by subtracting 273.15.
    - Winds (u, v) come from the 'ABI-L2-DMWVC' product (derived motion wind vectors) 
      for each time step.
    """
    # Generate half-hourly time ranges from 00:00 to 'hours' for the given date
    start = "00:00"
    end = str(hours).zfill(2) + ":00"
    time_list = tools.generate_time_ranges(date, minutes=30, start_time=start, end_time=end)
    total_times = len(time_list)

    # Download the MCMIP (multi-channel) GOES data for each time
    data_sets = []
    for i in range(total_times):
        ds = goes_nearesttime(
            time_list[i],
            satellite="noaa-goes16",
            product="ABI-L2-MCMIP",
            domain="C",
            return_as="xarray"
        )
        ds.expand_dims(dim={"time": [i]})  # Tag each dataset with a time index
        data_sets.append(ds)

    # Concatenate all downloaded datasets along the 'time' dimension
    data = xr.concat(data_sets, dim="time")

    # Keep only the desired band and the projection info
    data = data[[band, "goes_imager_projection"]]

    # Drop columns or rows with NaNs in the x dimension
    data = data.dropna(dim="x")

    # Downsample the data in the y and x dimensions
    data_full = data
    data = data.isel(y=slice(None, None, factor), x=slice(None, None, factor))

    # Extract the timestamps as fractional days
    time_array = data.t.values
    seconds_in_day = 24 * 60 * 60
    seconds_since_midnight = (time_array - time_array.astype('datetime64[D]')) / np.timedelta64(1, 's')
    T = torch.tensor(seconds_since_midnight / seconds_in_day, dtype=torch.float32)

    # Convert GOES x, y into Lambert-projected coordinates
    lonlat_full = calculate_degrees_goes(data_full)
    grid_full = torch.tensor(
        projections.Lambert.transform_points(
            projections.Geodetic, 
            lonlat_full[0],
            lonlat_full[1]
        )[:, :, 0:2],
        dtype=torch.float32
    )
    XY_full = torch.stack([grid_full[:, :, 0], grid_full[:, :, 1]], dim=-1).reshape(-1, 2)
    
    lonlat = calculate_degrees_goes(data)
    grid = torch.tensor(
        projections.Lambert.transform_points(
            projections.Geodetic, 
            lonlat[0],
            lonlat[1]
        )[:, :, 0:2],
        dtype=torch.float32
    )
    XY = torch.stack([grid[:, :, 0], grid[:, :, 1]], dim=-1).reshape(-1, 2)

    # Transform the extent bounds from lat/lon to Lambert x/y
    r = projections.Lambert.transform_point(extent[1], extent[3], projections.Geodetic)[0]
    l = projections.Lambert.transform_point(extent[0], extent[3], projections.Geodetic)[0]
    u = projections.Lambert.transform_point(extent[1], extent[3], projections.Geodetic)[1]
    d = projections.Lambert.transform_point(extent[0], extent[2], projections.Geodetic)[1]
    
    # Filter out points outside the bounding box
    mask_full = (XY_full[:, 0] >= l) & (XY_full[:, 0] <= r) & (XY_full[:, 1] >= d) & (XY_full[:, 1] <= u)
    XY_full = XY_full[mask_full]
    
    mask = (XY[:, 0] >= l) & (XY[:, 0] <= r) & (XY[:, 1] >= d) & (XY[:, 1] <= u)
    XY = XY[mask]

    # Extract brightness temperatures for the selected band, reshape, then subtract 273.15
    Z_full = torch.tensor(data_full[band].values,
        dtype=torch.float32
    ).reshape(total_times, -1) - 273.15
    
    Z = torch.tensor(data[band].values,
        dtype=torch.float32
    ).reshape(total_times, -1) - 273.15

    # Mask each time's brightness temperature
    Z_full = torch.stack([z[mask_full] for z in Z_full], dim=0)
    Z = torch.stack([z[mask] for z in Z], dim=0)

    # Prepare a list for wind vector data
    XYP_UV = []

    # Define bounding box in Lambert coords for wind data filtering
    lb = XY_full[:, 0].min().item()
    rb = XY_full[:, 0].max().item()
    db = XY_full[:, 1].min().item()
    ub = XY_full[:, 1].max().item()

    # Download wind vectors for each time step
    for i in range(len(time_list)):
        ds = goes_nearesttime(time_list[i], product='ABI-L2-DMWVC', return_as="xarray")
        ds = ds.dropna(dim="nMeasures")  # Remove points with NaNs
        
        # Extract wind speed/direction arrays
        wspd = ds.wind_speed.values
        wdir = ds.wind_direction.values
        wdir = np.deg2rad(wdir)  

        # Compute wind vectors u, v
        u_ = torch.tensor(-wspd * np.sin(wdir)).unsqueeze(1)
        v_ = torch.tensor(-wspd * np.cos(wdir)).unsqueeze(1)
        UV = torch.cat([u_, v_], dim=-1)

        # Convert lon/lat to Lambert for wind data
        lonlat = ds.lon.values, ds.lat.values
        P = torch.tensor(ds.pressure.values).unsqueeze(1)    
        XY_ = torch.tensor(
            projections.Lambert.transform_points(
                projections.Geodetic, 
                lonlat[0],
                lonlat[1]
            )[:, 0:2],
            dtype=torch.float32
        )

        # Filter to bounding box
        mask = (
            (XY_[:, 0] >= lb) & (XY_[:, 0] <= rb) &
            (XY_[:, 1] >= db) & (XY_[:, 1] <= ub)
        )

        XY_ = XY_[mask]
        UV = UV[mask]
        P = P[mask]
        XYP_UV.append(torch.cat([XY_, P, UV], dim=-1))

    data = model.data(T, XY, Z, XYP_UV, XY_full, Z_full)
    data.extent = extent
    data.date = date
    data.level = band
    return data