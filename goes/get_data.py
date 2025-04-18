import xarray as xr 
from goes2go.data import goes_timerange
import numpy as np
import torch 
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import Projection, Map, Model


def goes(Date, Hours, Band, Extent):

    start = Date +  " 00:00"
    end = Date + " " + str(Hours + 1).zfill(2) + ":00"

    files = goes_timerange(start, end,
                    satellite='goes16',
                    product='ABI-L2-CMIPC',
                    bands=Band,
                    return_as='filelist')["file"]
    data_sets = []
        
    for file in files:
        base = Path.cwd().parents[1] 
        path = base / 'data' / file
        data = xr.open_dataset(path)
        data_sets.append(data)
    
    
    time_array = np.array([data.t.values for data in data_sets])
    seconds_since_midnight = (time_array - time_array.astype('datetime64[D]')) / np.timedelta64(1, 's')
    T = torch.tensor(seconds_since_midnight, dtype=torch.float32).unsqueeze(1)

    data0 = xr.concat(data_sets, dim = "t")
    data0 = data0.dropna(dim="x")
    Geostationary = data0.metpy.parse_cf('CMI').metpy.cartopy_crs
    height = data0.goes_imager_projection.perspective_point_height
    x = torch.tensor(data0.x.values * height,dtype=torch.float32)
    y = torch.tensor(data0.y.values * height,dtype=torch.float32)
    X,Y = torch.meshgrid(x,y,indexing='xy')
    XY = torch.stack([X,Y], dim = -1).reshape(-1,2)
    
    r = Geostationary.transform_point(Extent[1], Extent[3], Projection.PlateCarree)[0]
    l = Geostationary.transform_point(Extent[0], Extent[3], Projection.PlateCarree)[0]
    u = Geostationary.transform_point(Extent[1], Extent[3], Projection.PlateCarree)[1]
    d = Geostationary.transform_point(Extent[0], Extent[2], Projection.PlateCarree)[1]
    mask = (XY[:, 0] >= l) & (XY[:, 0] <= r) & (XY[:, 1] >= d) & (XY[:, 1] <= u)
    XY = XY[mask]

    Z = []
    for i in range(len(data_sets)):
        Z.append(torch.tensor(data0.CMI.values[i], dtype=torch.float32).reshape(-1)[mask])
    
    Z = torch.stack(Z, dim = 0)
   
   
    product = 'ABI-L2-DMWC'
    if Band == 8:
        product = 'ABI-L2-DMWVC'
        
    files = goes_timerange(start, end,
                   satellite='goes16',
                   product=product,
                   bands=Band,
                   return_as='filelist')["file"]
   
    data_sets_vec = []
    XY_UV = []
    T_vec = []
    for file in files:
        base = Path.cwd().parents[1]
        path = base / 'data' / file
        data = xr.open_dataset(path)
        data = data.dropna(dim="nMeasures")
        data_sets_vec.append(data)
    pc = data_sets_vec[0].metpy.parse_cf('wind_speed').metpy.cartopy_crs

    
    for data in data_sets_vec:   
        
        lon, lat = data.lon.values, data.lat.values    
        XY_ = Geostationary.transform_points(pc, lon, lat)[:,0:2]
        XY_ = torch.tensor(XY_, dtype=torch.float32)
                
        wspd = data.wind_speed.values
        wdir = data.wind_direction.values
        wdir = np.deg2rad(wdir)  
        u_ = torch.tensor(-wspd * np.sin(wdir)).unsqueeze(1)
        v_ = torch.tensor(-wspd * np.cos(wdir)).unsqueeze(1)
        UV = torch.cat([u_, v_], dim=-1)
        
        mask = (XY_[:, 0] >= l) & (XY_[:, 0] <= r) & (XY_[:, 1] >= d) & (XY_[:, 1] <= u)
        XY_ = XY_[mask]
        UV = UV[mask]
        XY_UV.append(torch.cat([XY_, UV], dim=-1))
        
        
    time_array_vec = np.array([data.time.values[0] for data in data_sets_vec])
    seconds_since_midnight = (time_array_vec - time_array_vec.astype('datetime64[D]')) / np.timedelta64(1, 's')
    T_velocity = torch.tensor(seconds_since_midnight, dtype=torch.float32).unsqueeze(1)
    
    Data = Model.Data(T, XY, Z, XY_UV, T_velocity)
    Data.Map = Map.Parameters(Geostationary, Extent)
    Data.Date = Date
    Data.Level = Band
    Data.TemporalResolution = 5
    return Data