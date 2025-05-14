import torch
import cartopy
import pathlib
import xarray as xr
import numpy as np
import goes2go.data
import fields.coord_field
import fields.scalar_field
import fields.vector_field
from datetime import datetime

PlateCarree = cartopy.crs.PlateCarree()

def _download_band_paths(date, start, end, band):
    start = date + " " + start
    end = date + " " + end 
    paths = goes2go.data.goes_timerange(start, end,
                satellite='goes16',
                product='ABI-L2-CMIPC',
                bands=band,
                return_as='filelist')["file"]
    return paths

def _concat_datasets(paths):
    data_sets = []
    for path in paths:
        base = pathlib.Path.cwd().parents[0] 
        path = base / 'data' / path
        data = xr.open_dataset(path)
        data_sets.append(data)
    data0 = xr.concat(data_sets, dim="t")
    return data0

def _subset_dataset(ds, extent):
    height = ds.goes_imager_projection.perspective_point_height
    ds = ds.assign_coords(x=ds.x * height, y=ds.y * height)
    Geostationary = ds.metpy.parse_cf('CMI').metpy.cartopy_crs
    xmin, xmax, ymin, ymax = extent
    r = Geostationary.transform_point(xmax, ymax, PlateCarree)[0]
    l = Geostationary.transform_point(xmin, ymax, PlateCarree)[0]
    u = Geostationary.transform_point(xmax, ymax, PlateCarree)[1]
    d = Geostationary.transform_point(xmin, ymin, PlateCarree)[1]
    return ds.sel(x=slice(l, r), y=slice(u, d)), (l, r, d, u)

def _compute_locations(ds):
    x = torch.tensor(ds.x.values, dtype=torch.float32)
    y = torch.tensor(ds.y.values, dtype=torch.float32)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    return torch.stack([X, Y], dim=-1).reshape(-1, 2)

def _compute_scalar(ds):
    n = ds["CMI"].values.shape[0]
    return torch.tensor(ds["CMI"].values.copy(), dtype=torch.float32).reshape(n, -1)

def discrete_scalar_field(date, start, end, band, extent):
    paths = _download_band_paths(date, start, end, band)
    ds = _concat_datasets(paths)
    ds, extent = _subset_dataset(ds, extent)
    Geostationary = ds.metpy.parse_cf('CMI').metpy.cartopy_crs
    grid = ds["CMI"].values.shape[1:]
    time_array = ds.t.values
    seconds_since_midnight = (time_array - time_array.astype('datetime64[D]')) / np.timedelta64(1, 's')
    T = torch.tensor(seconds_since_midnight, dtype=torch.float32)
    XY = _compute_locations(ds)
    Z = _compute_scalar(ds)

    T = T
    Z = Z
    dcf = fields.coord_field.DiscreteCoordField(T, XY, Geostationary, extent, grid)
    dsf = fields.scalar_field.DiscreteScalarField(dcf, Z)
    return dsf

def _download_wind_data(date, time, band=9):
    datetime = f"{date} {time}"
    product = 'ABI-L2-DMWC'
    if band == 8:
        product = 'ABI-L2-DMWVC'
    file = goes2go.data.goes_nearesttime(
                attime=datetime,
                satellite='goes16',
                product=product,
                bands=band,
                return_as='filelist')["file"][0]
    base = pathlib.Path.cwd().parents[0]
    path = base / 'data' / file
    data = xr.open_dataset(path)
    data = data.dropna(dim="nMeasures")
    return data


def _compute_wind_func(ds, extent):
    paths = _download_band_paths("01-01-25", "00:00", "00:05", 9)
    dsb = _concat_datasets(paths)
    Geostationary = dsb.metpy.parse_cf('CMI').metpy.cartopy_crs
    pc = ds.metpy.parse_cf('wind_speed').metpy.cartopy_crs
    x = ds.lon.values
    y = ds.lat.values
    XY = Geostationary.transform_points(pc, x, y)[:, 0:2]
    XY = torch.tensor(XY, dtype=torch.float32)
    xmin, xmax, ymin, ymax = extent
    r = Geostationary.transform_point(xmax, ymax, pc)[0]
    l = Geostationary.transform_point(xmin, ymax, pc)[0]
    u = Geostationary.transform_point(xmax, ymax, pc)[1]
    d = Geostationary.transform_point(xmin, ymin, pc)[1]
    mask = (XY[:, 0] >= l) & (XY[:, 0] <= r) & (XY[:, 1] >= d) & (XY[:, 1] <= u)
    XY = XY[mask]
    

    XY = torch.stack([torch.tensor(x,dtype=torch.float32),
                    torch.tensor(y,dtype=torch.float32)], dim = -1)[mask]


    wspd = ds.wind_speed.values
    wdir = ds.wind_direction.values
    wdir = np.deg2rad(wdir)
    uu = torch.tensor(-wspd * np.sin(wdir)).unsqueeze(1)
    vv = torch.tensor(-wspd * np.cos(wdir)).unsqueeze(1)
    UV = torch.cat([uu, vv], dim=-1)[mask].unsqueeze(0)

    t = np.array([ds.time.values[0]])
    T = torch.tensor((t - t.astype('datetime64[D]')) / np.timedelta64(1, 's'), dtype=torch.float32)

    return T, XY, UV, PlateCarree, extent

def discrete_vector_field(date, time, band, extent):
    
    ds = _download_wind_data(date, time, band)
    T, XY, UV, PlateCarree, extent = _compute_wind_func(ds, extent)
    dcf = fields.coord_field.DiscreteCoordField(T, XY, PlateCarree, extent, None)
    dvf = fields.vector_field.DiscreteVectorField(dcf, UV)
    
    return dvf