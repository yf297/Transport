import xarray as xr 
import cartopy.crs as ccrs
from datetime import date, datetime, timedelta
import numpy as np
import random
import torch

from . import ode

Geodetic_proj = ccrs.Geodetic()

Lambert_proj = ccrs.LambertConformal(central_longitude=262.5, central_latitude=38.5,
                                standard_parallels=[38.5,38.5],
                                globe=ccrs.Globe(semimajor_axis=6371229, 
                                                 semiminor_axis=6371229))

def process_hrrr(path, i):
    ds = xr.open_dataset(path, engine='cfgrib')
    ds.expand_dims(dim={"time": [i]})
    return ds

def calculate_degrees_goes(data):
    x_coordinate_1d = data['x'][:] 
    y_coordinate_1d = data['y'][:]
    projection_info = data['goes_imager_projection']
    lon_origin = projection_info.longitude_of_projection_origin
    H = projection_info.perspective_point_height+projection_info.semi_major_axis
    r_eq = projection_info.semi_major_axis
    r_pol = projection_info.semi_minor_axis
    
    x_coordinate_2d, y_coordinate_2d = np.meshgrid(x_coordinate_1d, y_coordinate_1d)
    
    lambda_0 = (lon_origin*np.pi)/180.0  
    a_var = np.power(np.sin(x_coordinate_2d),2.0) + (np.power(np.cos(x_coordinate_2d),2.0)*(np.power(np.cos(y_coordinate_2d),2.0)+(((r_eq*r_eq)/(r_pol*r_pol))*np.power(np.sin(y_coordinate_2d),2.0))))
    b_var = -2.0*H*np.cos(x_coordinate_2d)*np.cos(y_coordinate_2d)
    c_var = (H**2.0)-(r_eq**2.0)
    r_s = (-1.0*b_var - np.sqrt((b_var**2)-(4.0*a_var*c_var)))/(2.0*a_var)
    s_x = r_s*np.cos(x_coordinate_2d)*np.cos(y_coordinate_2d)
    s_y = - r_s*np.sin(x_coordinate_2d)
    s_z = r_s*np.cos(x_coordinate_2d)*np.sin(y_coordinate_2d)
    
    np.seterr(all='ignore')
    
    abi_lat = (180.0/np.pi)*(np.arctan(((r_eq*r_eq)/(r_pol*r_pol))*((s_z/np.sqrt(((H-s_x)*(H-s_x))+(s_y*s_y))))))
    abi_lon = (lambda_0 - np.arctan(s_y/(H-s_x)))*(180.0/np.pi)
    
    return abi_lon[::-1], abi_lat[::-1]


def generate_time_ranges(date, minutes, hours):
    time = datetime.strptime(date + " 00:00", "%Y-%m-%d %H:%M")
    time_list = [time]
    for _ in range( ((60//minutes)  * hours) ):
        time = time + timedelta(minutes=minutes)
        time_list.append(time)
    return time_list


def generate_dates(n):
    start_date = date(2024, 4, 1)
    end_date = date(2024, 12, 30)

    all_dates = [
        (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range((end_date - start_date).days + 1)
    ]

    random.shuffle(all_dates)
    return all_dates[:n]

def rmse(data, scale=1, mag=1):
    T = data.T
    XY = data.XY
    n_frames = T.shape[0]
    n_points = XY.shape[0]

    # Initialize UV tensor
    UV = torch.zeros((n_frames, n_points, 2))

    # Compute velocity for each frame
    vel = ode.Vel_hat(data)
    for frame in range(n_frames):
        UV[frame] = vel(T[frame], XY) * scale * mag

    XY_UV = [torch.cat((XY, UV[frame]), dim=-1).detach() for frame in range(n_frames)]

    errors = []
    for i in range(data.n):
        true_U = data.XY_UV[i][:, 2]
        true_V = data.XY_UV[i][:, 3]
        pred_U = XY_UV[i][:, 2]
        pred_V = XY_UV[i][:, 3]

        mse_U = torch.mean((true_U - pred_U) ** 2)
        mse_V = torch.mean((true_V - pred_V) ** 2)
        rmse = torch.sqrt(mse_U + mse_V)
        errors.append(rmse)

    mean_rmse = torch.stack(errors).mean()

    return mean_rmse


def rmse_U(data, scale=1, mag=1):
    T = data.T
    XY = data.XY
    n_frames = T.shape[0]
    n_points = XY.shape[0]

    # Initialize UV tensor
    UV = torch.zeros((n_frames, n_points, 2))

    # Compute velocity for each frame
    vel = ode.Vel_hat(data)
    for frame in range(n_frames):
        UV[frame] = vel(T[frame], XY) * scale * mag

    XY_UV = [torch.cat((XY, UV[frame]), dim=-1).detach() for frame in range(n_frames)]

    errors = []
    for i in range(data.n):
        true_U = data.XY_UV[i][:, 2]
        pred_U = XY_UV[i][:, 2]

        mse_U = torch.mean((true_U - pred_U) ** 2)
        rmse = torch.sqrt(mse_U)
        errors.append(rmse)

    mean_rmse = torch.stack(errors).mean()

    return mean_rmse

def rmse_V(data, scale=1, mag=1):
    T = data.T
    XY = data.XY
    n_frames = T.shape[0]
    n_points = XY.shape[0]

    UV = torch.zeros((n_frames, n_points, 2))

    vel = ode.Vel_hat(data)
    for frame in range(n_frames):
        UV[frame] = vel(T[frame], XY) * scale * mag

    XY_UV = [torch.cat((XY, UV[frame]), dim=-1).detach() for frame in range(n_frames)]

    errors = []
    for i in range(data.n):
        true_V = data.XY_UV[i][:, 3]
        pred_V = XY_UV[i][:, 3]

        mse_V = torch.mean((true_V - pred_V) ** 2)
        rmse = torch.sqrt(mse_V)
        errors.append(rmse)

    mean_rmse = torch.stack(errors).mean()

    return mean_rmse


def point_sampling(points, min_dist, max_samples=1000):
    n = points.shape[0]
    perm = torch.randperm(n)  
    chosen = [perm[0]]  
    idx = 1

    while idx < n and len(chosen) < max_samples:
        candidate_idx = perm[idx]
        candidate = points[candidate_idx]

        # Get selected points
        chosen_points = points[torch.tensor(chosen)]

        # Check if candidate satisfies min_dist constraint
        dists = torch.norm(candidate - chosen_points, dim=1)
        
        if torch.all(dists >= min_dist):
            chosen.append(candidate_idx)

        idx += 1  # Move to the next candidate

    return torch.tensor(chosen)