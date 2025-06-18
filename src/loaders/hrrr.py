import torch
import cartopy
import pathlib
import xarray as xr
import herbie
import fields.discrete

PlateCarree = cartopy.crs.PlateCarree()
Lambert = cartopy.crs.LambertConformal(central_longitude=262.5, central_latitude=38.5,
                                standard_parallels=[38.5,38.5],
                                globe=cartopy.crs.Globe(semimajor_axis=6371229, 
                                                 semiminor_axis=6371229))



class Data:
    def __init__(self, date, hours, level):
        self.date = date
        self.hours = hours
        self.level = level
        self.ds = self._download_and_open()

    def _download_and_open(self):
        paths = []
        for fxx in range(self.hours + 1):
            p = herbie.Herbie(self.date, model="hrrr", fxx=fxx).download(f"{self.level} mb")
            paths.append(pathlib.Path(p))
        parts = []
        for i, p in enumerate(paths):
            ds = xr.open_dataset(p, engine="cfgrib")
            ds = ds.expand_dims(time=[i])
            parts.append(ds)
        ds = xr.concat(parts, dim="time")
        ds = ds.assign_coords(longitude=(ds.longitude - 360))
        ds = ds.isel(y=slice(None, None, -1))
        return ds

    def _subset(self, extent):
        ds = self.ds
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
        ds = ds.sel(x=slice(l, r, 4), y=slice(u, d, 4))
        return ds, (l, r, d, u)

    def _compute_locations(self, ds):
        x = torch.tensor(ds.x.values, dtype=torch.float32)
        y = torch.tensor(ds.y.values, dtype=torch.float32)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        return torch.stack([X, Y], dim=-1).reshape(-1, 2)

    def _compute_scalar(self, ds):
        return torch.tensor(ds.dpt.values.copy(), dtype=torch.float32).reshape(-1) 

    def _compute_vector(self, ds):
        u = torch.tensor(ds.u.values.copy(), dtype=torch.float32).view(-1,1)
        v = torch.tensor(ds.v.values.copy(), dtype=torch.float32).view(-1,1)
        return torch.cat([u, v], dim=-1)

    def scalar_field(self, extent):
        ds, extent2 = self._subset(extent)
        grid = ds.dpt.shape
        T = torch.linspace(0, self.hours * 3600, self.hours + 1)
        T = T.unsqueeze(1)
        XY = self._compute_locations(ds)
        TXY = torch.cat([T.repeat_interleave(XY.size(0)).unsqueeze(-1), 
                         XY.repeat(T.size(0), 1)], dim=-1)
        dcf = fields.discrete.CoordField(TXY, Lambert, extent2, grid)
        Z = self._compute_scalar(ds)
        return fields.discrete.ScalarField(dcf, Z)

    def vector_field(self, extent):
        ds, extent2 = self._subset(extent)
        grid = ds.dpt.shape
        T = torch.linspace(0, self.hours * 3600, self.hours + 1)
        T = T.unsqueeze(1)
        XY = self._compute_locations(ds)
        TXY = torch.cat([T.repeat_interleave(XY.size(0)).unsqueeze(-1), 
                         XY.repeat(T.size(0), 1)], dim=-1)
        dcf = fields.discrete.CoordField(TXY, Lambert, extent2, grid)
        UV = self._compute_vector(ds)
        return fields.discrete.VectorField(dcf, UV)