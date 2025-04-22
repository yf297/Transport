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
from typing import Tuple



PlateCarree = cartopy.crs.PlateCarree()

def _download_band_paths(date: str, 
                    start: str, 
                    end: str,
                    band: int = 9
) -> list[pathlib.Path]:
    start = date + " " + start
    end = date + " " +  end 
    paths = goes2go.data.goes_timerange(start, end,
                satellite='goes16',
                product='ABI-L2-CMIPC',
                bands=band,
                return_as='filelist')["file"]
    return paths

def _concat_datasets(
    paths: list[pathlib.Path]
) -> xr.Dataset:
    data_sets = []
    for path in paths:
        base = pathlib.Path.cwd().parents[0] 
        path = base / 'data' / path
        data = xr.open_dataset(path)
        data_sets.append(data)
    data0 = xr.concat(data_sets, dim = "t")
    return data0


def _subset_dataset(
    ds: xr.Dataset,
    extent: Tuple[float, float, float, float]
) -> Tuple[xr.Dataset, Tuple[float, float, float, float]]:
    height = ds.goes_imager_projection.perspective_point_height
    ds = ds.assign_coords(x=ds.x * height, y=ds.y * height )
    Geostationary = ds.metpy.parse_cf('CMI').metpy.cartopy_crs
    xmin, xmax, ymin, ymax = extent
    r = Geostationary.transform_point(xmax, ymax, PlateCarree)[0]
    l = Geostationary.transform_point(xmin, ymax, PlateCarree)[0]
    u = Geostationary.transform_point(xmax, ymax, PlateCarree)[1]
    d = Geostationary.transform_point(xmin, ymin, PlateCarree)[1]
    ds = ds.isel(x=slice(None, None, 3), y=slice(None, None, 3))
    return ds.sel(x=slice(l, r), y=slice(u, d)), (l, r, d, u)

def _compute_locations(
    ds: xr.Dataset
) -> torch.Tensor:
    x = torch.tensor(ds.x.values, dtype=torch.float32)
    y = torch.tensor(ds.y.values, dtype=torch.float32)
    X,Y = torch.meshgrid(x,y,indexing='xy')
    return torch.stack([X,Y], dim = -1)

def _compute_scalar(
    ds: xr.Dataset
) -> torch.Tensor:
    return torch.tensor(ds["CMI"].values.copy(), dtype=torch.float32)

def discrete_scalar_field(
    date: str, 
    start: str, 
    end: str,
    by:str,
    band: int,
    extent: Tuple[float, float, float, float]
) -> fields.scalar_field.DiscreteScalarField:
    paths = _download_band_paths(date, start, end, band)
    ds = _concat_datasets(paths)
    ds, extent = _subset_dataset(ds, extent)
    Geostationary = ds.metpy.parse_cf('CMI').metpy.cartopy_crs
    
    time_array = ds.t.values
    seconds_since_midnight = (time_array - time_array.astype('datetime64[D]')) / np.timedelta64(1, 's')
    times = torch.tensor(seconds_since_midnight, dtype=torch.float32).unsqueeze(1)
    locations = _compute_locations(ds)
    scalar = _compute_scalar(ds)

    h, m = map(int, by.split(":"))
    step = (h * 60 + m)//5
    times = times[::step]
    scalar = scalar[::step, :,:]
    dcf = fields.coord_field.DiscreteCoordField(times, locations, Geostationary, extent)
    dsf = fields.scalar_field.DiscreteScalarField(dcf, scalar)
    return dsf

