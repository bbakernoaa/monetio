import os
import numpy as np
from pyproj import Proj, CRS

path = os.path.abspath(__file__)


class MockArea:
    def __init__(self, proj_dict, area_extent, nx, ny):
        self.proj_dict = proj_dict
        self.area_extent = area_extent
        self.nx = nx
        self.ny = ny
        self.proj_str = Proj(proj_dict).srs

    def get_lonlats(self):
        x = np.linspace(self.area_extent[0], self.area_extent[2], self.nx)
        y = np.linspace(self.area_extent[1], self.area_extent[3], self.ny)
        xv, yv = np.meshgrid(x, y)
        p = Proj(self.proj_dict)
        return p(xv, yv, inverse=True)

    def get_lonlats_dask(self):
        try:
            import dask.array as da

            x = da.linspace(self.area_extent[0], self.area_extent[2], self.nx)
            y = da.linspace(self.area_extent[1], self.area_extent[3], self.ny)
            xv, yv = da.meshgrid(x, y)

            def _proj_inv(x_val, y_val):
                p = Proj(self.proj_dict)
                return np.stack(p(x_val, y_val, inverse=True))

            # Use map_blocks to keep it lazy
            combined = da.map_blocks(_proj_inv, xv, yv, dtype=float, chunks=(2, *xv.chunks))
            return combined[0], combined[1]
        except ImportError:
            return self.get_lonlats()

    def to_cartopy_crs(self):
        try:
            import cartopy.crs as ccrs

            # Use ccrs.Proj to wrap a Proj string, which is a concrete class.
            return ccrs.Proj(CRS.from_user_input(self.proj_dict).to_proj4())
        except ImportError:
            return None


def _geos_16_grid(dset):
    projection = dset.goes_imager_projection
    h = projection.perspective_point_height
    a = projection.semi_major_axis
    b = projection.semi_minor_axis
    lon_0 = projection.longitude_of_projection_origin
    sweep = projection.sweep_angle_axis
    x = dset.x * h
    y = dset.y * h
    x_ll = x[0]
    x_ur = x[-1]
    y_ll = y[0]
    y_ur = y[-1]
    x_h = (x_ur - x_ll) / (len(x) - 1.0) / 2.0
    y_h = (y_ur - y_ll) / (len(y) - 1.0) / 2.0
    area_extent = (x_ll - x_h, y_ll - y_h, x_ur + x_h, y_ur + y_h)

    proj_dict = {
        "a": float(a),
        "b": float(b),
        "lon_0": float(lon_0),
        "h": float(h),
        "proj": "geos",
        "units": "m",
        "sweep": sweep,
    }
    return MockArea(proj_dict, area_extent, len(dset.x), len(dset.y))


def _get_sinu_grid_df():
    from pandas import read_csv

    f = os.path.join(os.path.dirname(path), "data/sn_bound_10deg.txt")
    td = read_csv(f, skiprows=4, sep=r"\s+")
    td = td.assign(
        ihiv="h" + td.ih.astype(str).str.zfill(2) + "v" + td.iv.astype(str).str.zfill(2)
    )
    return td


def _sinu_grid_latlon_boundary(h, v):
    td = _get_sinu_grid_df()
    o = td.loc[(td.ih == int(h)) & (td.iv == int(v))]
    latmin = o.lat_min.iloc[0]
    lonmin = o.lon_min.iloc[0]
    latmax = o.lat_max.iloc[0]
    lonmax = o.lon_max.iloc[0]
    return lonmin, latmin, lonmax, latmax


def _get_sinu_xy(lon, lat):
    sinu = Proj(
        "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m"
    )
    return sinu(lon, lat)


def _get_sinu_latlon(x, y):
    xv, yv = np.meshgrid(x, y)
    sinu = Proj(
        "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +R=6371007.181"
    )
    return sinu(xv, yv, inverse=True)


def get_sinu_area_extent(lonmin, latmin, lonmax, latmax):
    xmin, ymin = _get_sinu_xy(lonmin, latmin)
    xmax, ymax = _get_sinu_xy(lonmax, latmax)
    return (xmin, ymin, xmax, ymax)


def get_modis_latlon_from_swath_hv(h, v, dset):
    lonmin, latmin, lonmax, latmax = _sinu_grid_latlon_boundary(h, v)
    xmin, ymin = _get_sinu_xy(lonmin, latmin)
    xmax, ymax = _get_sinu_xy(lonmax, latmax)
    x = np.linspace(xmin, xmax, len(dset.x))
    y = np.linspace(ymin, ymax, len(dset.y))
    lon, lat = _get_sinu_latlon(x, y)
    dset.coords["longitude"] = (("x", "y"), lon)
    dset.coords["latitude"] = (("x", "y"), lat)
    dset.attrs["area_extent"] = (x.min(), y.min(), x.max(), y.max())
    dset.attrs["proj4_srs"] = (
        "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m"
    )
    return dset


def get_sinu_area_def(dset):
    proj4_srs = dset.attrs["proj4_srs"]
    area_extent = dset.attrs["area_extent"]
    nx, ny = dset.longitude.shape
    return MockArea(proj4_srs, area_extent, nx, ny)


def get_ioapi_pyresample_area_def(ds, proj4_srs):
    x_ll, y_ll = ds.XORIG + ds.XCELL * 0.5, ds.YORIG + ds.YCELL * 0.5
    x_ur, y_ur = (
        ds.XORIG + (ds.NCOLS * ds.XCELL) + 0.5 * ds.XCELL,
        ds.YORIG + (ds.YCELL * ds.NROWS) + 0.5 * ds.YCELL,
    )
    area_extent = (x_ll, y_ll, x_ur, y_ur)
    return MockArea(proj4_srs, area_extent, ds.NCOLS, ds.NROWS)


def get_generic_projection_from_proj4(lat, lon, proj4_srs):
    # This used to compute optimal BB area. Without pyresample, we just return CRS.
    return CRS.from_user_input(proj4_srs)


def get_optimal_cartopy_proj(lat, lon, proj4_srs):
    try:
        import cartopy.crs as ccrs

        # Use ccrs.Proj to wrap a Proj string, which is a concrete class.
        return ccrs.Proj(CRS.from_user_input(proj4_srs).to_proj4())
    except ImportError:
        return None


def _ioapi_grid_from_dataset(ds, earth_radius=6370000):
    pargs = dict()
    pargs["lat_1"] = ds.P_ALP
    pargs["lat_2"] = ds.P_BET
    pargs["lat_0"] = ds.YCENT
    pargs["lon_0"] = ds.P_GAM
    pargs["center_lon"] = ds.XCENT
    pargs["x0"] = ds.XORIG
    pargs["y0"] = ds.YORIG
    pargs["r"] = earth_radius
    proj_id = ds.GDTYP
    if proj_id == 2:
        p4 = "+proj=lcc +lat_1={lat_1} +lat_2={lat_2} +lat_0={lat_0} +lon_0={lon_0} +x_0=0 +y_0=0 +datum=WGS84 +units=m +a={r} +b={r}"
        p4 = p4.format(**pargs)
    elif proj_id == 4:
        p4 = "+proj=stere +lat_ts={lat_1} +lon_0={lon_0} +lat_0=90.0 +x_0=0 +y_0=0 +a={r} +b={r}"
        p4 = p4.format(**pargs)
    elif proj_id == 3:
        p4 = "+proj=merc +lat_ts={lat_1} +lon_0={center_lon} +x_0={x0} +y_0={y0} +a={r} +b={r}"
        p4 = p4.format(**pargs)
    else:
        raise NotImplementedError(f"IOAPI proj not implemented yet: {proj_id}")
    return p4


def get_latlon_ioapi(dset, proj4_srs):
    x = np.linspace(
        dset.XORIG + dset.XCELL * 0.5,
        dset.XORIG + (dset.NCOLS - 0.5) * dset.XCELL,
        dset.NCOLS,
    )
    y = np.linspace(
        dset.YORIG + dset.YCELL * 0.5,
        dset.YORIG + (dset.NROWS - 0.5) * dset.YCELL,
        dset.NROWS,
    )
    xv, yv = np.meshgrid(x, y)
    p = Proj(proj4_srs)
    lon, lat = p(xv, yv, inverse=True)
    return lon, lat


def grid_from_dataset(ds, earth_radius=6370000):
    if hasattr(ds, "IOAPI_VERSION") or hasattr(ds, "P_ALP"):
        return _ioapi_grid_from_dataset(ds, earth_radius=earth_radius)
    return None
