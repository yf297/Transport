from pathlib import Path
from typing import Tuple

import xarray as xr
import torch
from herbie import Herbie

from geo.proj import Lambert, PlateCarree
from geo.coords import Map
from fields.scalar_field import ScalarField
from fields.vector_field import VectorField


def _download_paths(date: str, hours: int, level: str) -> list[Path]:
    paths: list[Path] = []
    for fxx in range(hours + 1):
        p = Herbie(date, model="hrrr", fxx=fxx).download(level)
        paths.append(Path(p))
    return paths


def _open_and_tag(ds: xr.Dataset, i: int) -> xr.Dataset:
    return ds.expand_dims(time=[i])


def _concat_datasets(paths: list[Path]) -> xr.Dataset:
    parts: list[xr.Dataset] = []
    for i, p in enumerate(paths):
        ds = xr.open_dataset(p, engine="cfgrib")
        parts.append(_open_and_tag(ds, i))

    ds = xr.concat(parts, dim="time")
    ds = ds.assign_coords(longitude=(ds.longitude - 360))
    ds = ds.isel(y=slice(None, None, -1))
    return ds


def _subset_dataset(
    ds: xr.Dataset,
    extent: Tuple[float, float, float, float]
) -> Tuple[xr.Dataset, Tuple[float, float, float, float]]:
    lon = ds.longitude.values
    lat = ds.latitude.values

    points = Lambert.transform_points(PlateCarree, lon, lat)
    locations = points[:, :, :2]

    ds = ds.assign_coords(x=("x", locations[0, :, 0]))
    ds = ds.assign_coords(y=("y", locations[:, 0, 1]))

    xmin, xmax, ymin, ymax = extent
    r = Lambert.transform_point(xmax, ymax, PlateCarree)[0]
    l = Lambert.transform_point(xmin, ymax, PlateCarree)[0]
    u = Lambert.transform_point(xmax, ymax, PlateCarree)[1]
    d = Lambert.transform_point(xmin, ymin, PlateCarree)[1]
    return ds.sel(x=slice(l, r), y=slice(u, d)), (l, r, d, u)


def _compute_locations(ds: xr.Dataset) -> torch.Tensor:
    x = torch.tensor(ds.x.values, dtype=torch.float32)
    y = torch.tensor(ds.y.values, dtype=torch.float32)
    X,Y = torch.meshgrid(x,y,indexing='xy')
    return torch.stack([X,Y], dim = -1)


def _compute_scalar(ds: xr.Dataset) -> torch.Tensor:
    return torch.tensor(ds.dpt.values.copy(), dtype=torch.float32)


def _compute_vector(ds: xr.Dataset) -> torch.Tensor:
    u = torch.tensor(ds.u.values.copy(), dtype=torch.float32)
    v = torch.tensor(ds.v.values.copy(), dtype=torch.float32)
    return torch.stack([u, v], dim=-1)


def load_scalar_field(
    date: str,
    hours: int,
    level: int,
    extent: Tuple[float, float, float, float]
) -> ScalarField:
    paths = _download_paths(date, hours, level)
    ds = _concat_datasets(paths)
    ds, extent = _subset_dataset(ds, extent)

    times = torch.linspace(0, hours * 3600, hours + 1)
    locations = _compute_locations(ds)
    scalar = _compute_scalar(ds)

    scalar_field = ScalarField(times, locations, scalar)
    scalar_field.map = Map(Lambert, extent)
    return scalar_field


def load_vector_field(
    date: str,
    hours: int,
    level: int,
    extent: Tuple[float, float, float, float]
) -> VectorField:
    paths = _download_paths(date, hours, level)
    ds = _concat_datasets(paths)
    ds, extent = _subset_dataset(ds, extent)

    times = torch.linspace(0, hours * 3600, hours + 1)
    locations = _compute_locations(ds)
    vector = _compute_vector(ds)

    vector_field = VectorField(times, locations, vector)
    vector_field.map = Map(Lambert, extent)
    return vector_field