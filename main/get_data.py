from herbie import Herbie
import xarray as xr 
import cartopy.crs as ccrs
import torch 
from goes2go.data import  goes_nearesttime
import numpy as np

from . import tools

geo = ccrs.Geostationary(central_longitude=-137.0, satellite_height=35786023.0)

def hrrr(date, level, hours, extent, factor):
    paths = []
    
    start = "00:00"
    end  = str(hours).zfill(2) + ":00"
    time_list = tools.generate_time_ranges(date,
                                           minutes = 60, 
                                           start_time = start, 
                                           end_time = end)
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
    data = data.isel(y=slice(None, None, factor), x=slice(None, None, factor))

    
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



def goes(date, 
         minutes,
         center,
         band, 
         extent,
         factor):
    
    data_sets = []
    start, end  = tools.generate_balanced_times(minutes, center)
    
    time_list = tools.generate_time_ranges(date, 
                                           minutes, 
                                           start_time= start,
                                           end_time = end)
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
    
    data = data[[band,"goes_imager_projection"]]
    data = data.dropna(dim = "x")
    
    data = data.isel(y=slice(None, None, factor), x=slice(None, None, factor))
    
    hour, minutes = map(int, start.split(":"))
    hour_start =  hour + minutes / 60
    hour, minutes = map(int, end.split(":"))
    hour_end =  hour + minutes / 60
    T = torch.linspace(hour_start/24, hour_end/24, total_times)
    
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
    
    Z = torch.tensor(np.flip(data[band].values, axis = 1).copy(),
                     dtype=torch.float32).reshape(total_times,-1) - 273.15

    
    Z = [z[mask] for z in Z]
    
    XY_UV = []
    lb = XY[:, 0].min().item()
    rb = XY[:, 0].max().item()
    db = XY[:, 1].min().item()
    ub = XY[:, 1].max().item()

    
    for i in range(0,len(time_list)):
        ds = goes_nearesttime(time_list[i], 
                                    product='ABI-L2-DMWVC',
                                    return_as="xarray")
        ds = ds.dropna(dim = "nMeasures")
        wspd = ds.wind_speed.values
        wdir = ds.wind_direction.values
        wdir = np.deg2rad(wdir)

        u = torch.tensor(-wspd * np.sin(wdir)).unsqueeze(1)
        v = torch.tensor(-wspd * np.cos(wdir)).unsqueeze(1)
        UV = torch.cat([u,v], dim = -1)
        lonlat = ds.lon.values, ds.lat.values[::-1]
        
        XY_ = torch.tensor(tools.Lambert_proj.transform_points(
        ccrs.Geodetic(), 
        lonlat[0],
        lonlat[1])[:,0:2],dtype=torch.float32)
        
        mask = (
            (XY_[:, 0] >= lb) & (XY_[:, 0] <= rb) &
            (XY_[:, 1] >= db) & (XY_[:, 1] <= ub)
        )
        
        XY_ = XY_[mask]
        UV = torch.cat([u,v],dim = -1)
        UV = UV[mask]
        XY_UV.append(torch.cat([XY_,UV],
                dim = -1))
        
    return T, XY, Z, XY_UV