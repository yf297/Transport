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
    start_date = date(2024, 1, 1)
    end_date = date(2024, 12, 31)
    delta = (end_date - start_date).days

    dates = []
    for _ in range(n):
        days_offset = random.randint(0, delta)
        random_date = start_date + timedelta(days=days_offset)
        dates.append(random_date.strftime("%Y-%m-%d"))
    return dates


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


def rmse(data, scale = 1, mag = 1):
    T = data.T
    XY = data.XY
    
    UV = torch.ones((T.shape[0], XY.shape[0], 2 ))
    
    vel = ode.Vel_hat(data)
    for frame in range(0,T.shape[0]):
        UV[frame,:,:] = vel(T[frame], XY) * scale * mag
        
    XY_UV = [torch.cat([XY,
                        UV[i,:,0:1],
                        UV[i,:,1:2]],
            dim = -1).detach() for i in range(0,T.shape[0])]
        
    errors =  [
            torch.sqrt(torch.sum((data.XY_UV[i][:,2] - XY_UV[i][:,2])**2) / data.m + 
            torch.sum((data.XY_UV[i][:,3] - XY_UV[i][:,3])**2) / data.m)
            
            for i in range(data.n)
            ]   
    return torch.stack(errors).mean()