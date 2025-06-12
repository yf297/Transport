import torch
import cartopy
import pathlib
import xarray as xr
import numpy as np
import goes2go.data
import fields.discrete

PlateCarree = cartopy.crs.PlateCarree()

class Data:
    def __init__(self, start, end, time, band, factor = 1):
        self.start = start
        self.end = end
        self.time = time
        self.factor = factor
        
        self.band = band
        self.scalar_ds = self.form_scalar_df()
        self.vector_ds = self.form_vector_df()

    def form_scalar_df(self):
        paths = goes2go.data.goes_timerange(
                self.start, self.end,
                satellite='goes16',
                product='ABI-L2-CMIPC',
                bands=self.band,
                return_as='filelist'
        )["file"]
                
        data_sets = []
        for path in paths:
            base = pathlib.Path.cwd().parents[0]
            path = base / 'data' / path
            data = xr.open_dataset(path)
            data_sets.append(data)
        scalar_ds = xr.concat(data_sets, dim="t")
        height = scalar_ds.goes_imager_projection.perspective_point_height
        scalar_ds = scalar_ds.assign_coords(x=scalar_ds.x * height, y=scalar_ds.y * height)
        return scalar_ds


    def scalar_field(self, extent):
        xmin, xmax, ymin, ymax = extent
        geo = self.scalar_ds.metpy.parse_cf('CMI').metpy.cartopy_crs
        self.geo = geo
        r = geo.transform_point(xmax + 0.5, ymax + 0.5, PlateCarree)[0]
        l = geo.transform_point(xmin - 1.5, ymax + 0.5, PlateCarree)[0]
        u = geo.transform_point(xmax + 0.5, ymax + 0.5, PlateCarree)[1]
        d = geo.transform_point(xmin - 1, ymin - 0.5, PlateCarree)[1]
        
        ds = self.scalar_ds.sel(x=slice(l, r, self.factor), y=slice(u, d, self.factor))
        extent = (l, r, d, u)
        
        grid = ds["CMI"].values.shape[1:]
        time_array = ds.t.values
        seconds_since_midnight = (time_array - time_array.astype('datetime64[D]')) / np.timedelta64(1, 's')
        T = torch.tensor(seconds_since_midnight, dtype=torch.float32)
        T = T.unsqueeze(1)
        x = torch.tensor(ds.x.values, dtype=torch.float32)
        y = torch.tensor(ds.y.values, dtype=torch.float32)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        XY = torch.stack([X, Y], dim=-1).reshape(-1, 2)
        
        n = ds["CMI"].values.shape[0]
        Z = torch.tensor(ds["CMI"].values.copy(), dtype=torch.float32).reshape(n, -1) 

        geo = ds.metpy.parse_cf('CMI').metpy.cartopy_crs
        dcf = fields.discrete.CoordField(T, XY, geo, extent, grid)
        dsf = fields.discrete.ScalarField(dcf, Z)
        return dsf

    def form_vector_df(self):
        product = 'ABI-L2-DMWC'
        if self.band == 8:
            product = 'ABI-L2-DMWVC'
        file = goes2go.data.goes_nearesttime(
            attime=self.time,
            satellite='goes16',
            product=product,
            bands=self.band,
            return_as='filelist'
        )["file"][0]
        base = pathlib.Path.cwd().parents[0]
        path = base / 'data' / file
        vector_ds = xr.open_dataset(path)
        vector_ds = vector_ds.dropna(dim="nMeasures")
        
        return vector_ds


    def vector_field(self, extent):
        l, r, d, u = extent
        sel = ((self.vector_ds['lon'] >= l) & (self.vector_ds['lon'] <= r) & (self.vector_ds['lat'] >= d) & (self.vector_ds['lat'] <= u))
        ds = self.vector_ds.isel(nMeasures=sel)
        pc = ds.metpy.parse_cf('wind_speed').metpy.cartopy_crs
        self.pc = pc
        t = np.array([ds.time.values[0]])
        T = torch.tensor((t - t.astype('datetime64[D]')) / np.timedelta64(1, 's'), dtype=torch.float32)
        T = T.unsqueeze(1)
        
        x = torch.tensor(ds.lon.values, dtype=torch.float32)
        y = torch.tensor(ds.lat.values, dtype=torch.float32)
        XY = XY = torch.stack([x, y], dim=-1) 

        wspd = ds.wind_speed.values
        wdir = ds.wind_direction.values
        wdir = np.deg2rad(wdir)
        uu = torch.tensor(-wspd * np.sin(wdir)).unsqueeze(1)
        vv = torch.tensor(-wspd * np.cos(wdir)).unsqueeze(1)
        UV = torch.cat([uu, vv], dim=-1).unsqueeze(0)

        dcf = fields.discrete.CoordField(T, XY, pc, extent, None)
        dvf = fields.discrete.VectorField(dcf, UV)
        return dvf

