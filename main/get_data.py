from herbie import Herbie
import xarray as xr 
import cartopy.crs as ccrs
import torch 
from goes2go.data import  goes_nearesttime
import numpy as np

from . import tools

geo = ccrs.Geostationary(central_longitude=-137.0, satellite_height=35786023.0)

def hrrr(date, level, hours, extent):
    paths = []
    time_list = tools.generate_time_ranges(date, minutes = 60, hours = hours)
    total_times = len(time_list)
    for i in range(0,total_times):
        paths.append(Herbie(date, model="hrrr", fxx=i).download(level))
        
    data = xr.concat([tools.process_hrrr(paths[i], i) for i in range(0,total_times)], 
                    dim='time')

    data["longitude"] = data["longitude"] - 360
    
    
    grid = tools.Lambert_proj.transform_points(
    	ccrs.Geodetic(), 
    	data.longitude.values,
    	data.latitude.values)[:,:,0:2]
    
    data = data.assign_coords(x=("x",  grid[0,:,0]))
    data = data.assign_coords(y=("y",  grid[:,0,1]))
    
    (l,d) = tools.Lambert_proj.transform_point(extent[0],extent[2],ccrs.Geodetic())
    (r,u) = tools.Lambert_proj.transform_point(extent[1],extent[3],ccrs.Geodetic())
    
    data = data.sel(x = slice(l,r), y = slice(d,u))
    
    Z = torch.tensor(data.dpt.values, dtype=torch.float32).reshape(total_times,-1) - 273.15
    XY = torch.tensor(tools.Lambert_proj.transform_points(
        ccrs.Geodetic(), 
        data.longitude.values,
        data.latitude.values)[:,:,0:2], dtype=torch.float32).reshape(-1,2)
    
    T = torch.linspace(0, hours/24, total_times)
    XY_UV = [torch.cat([XY,
                        torch.tensor(data.u.values.reshape(total_times,-1,1)[i]),
                        torch.tensor(data.v.values.reshape(total_times,-1,1)[i])],
                     dim = -1) for i in range(0,len(T))]
    
    return T, XY, Z, XY_UV






def goes(date, minutes, hours, extent):
    data_sets = []
    time_list = tools.generate_time_ranges(date, minutes, hours)
    total_times = len(time_list)
    for i in range(0,total_times):
        ds = goes_nearesttime(time_list[i], 
                                    satellite="noaa-goes16", 
                                    product="ABI-L2-MCMIP", 
                                    domain = "C",
                                    return_as="xarray")
        ds.expand_dims(dim={"time": [i]})
        data_sets.append(ds)

    data = xr.concat(data_sets, dim="time")
    data = data[["CMI_C08","CMI_C09","CMI_C10","goes_imager_projection"]]
    data = data.dropna(dim = "x")
    
    T = torch.linspace(0, hours/24, total_times)
    
    lonlat = tools.calculate_degrees_goes(data)
    grid = torch.tensor(tools.Lambert_proj.transform_points(
        ccrs.Geodetic(), 
        lonlat[0],
        lonlat[1])[:,:,0:2], dtype=torch.float32)
    
    XY = torch.stack([grid[:,:,0],grid[:,:,1]], dim = -1).reshape(-1,2)
    
    (l,d) = tools.Lambert_proj.transform_point(extent[0],extent[2],ccrs.Geodetic())
    (r,u) = tools.Lambert_proj.transform_point(extent[1],extent[3],ccrs.Geodetic())
    
    mask = (XY[:, 0] >= l) & (XY[:, 0] <= r) & (XY[:, 1] >= d) & (XY[:, 1] <= u)
    XY = XY[mask]
    
    Z08 = torch.tensor(np.flip(data.CMI_C08.values, axis = 1).copy(),
                     dtype=torch.float32).reshape(total_times,-1) - 273.15
    Z09 = torch.tensor(np.flip(data.CMI_C09.values, axis = 1).copy(),
                     dtype=torch.float32).reshape(total_times,-1) - 273.15
    Z10 = torch.tensor(np.flip(data.CMI_C10.values, axis = 1).copy(),
                    dtype=torch.float32).reshape(total_times,-1) - 273.15
    
    Z08 = [Z[mask] for Z in Z08]
    Z09 = [Z[mask] for Z in Z09]
    Z10 = [Z[mask] for Z in Z10]
    
    XY_UV = []

    
    return T, XY, [Z08, Z09, Z10], XY_UV


 