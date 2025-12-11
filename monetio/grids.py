import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

# Define the data directory relative to this file
DATA_DIR = Path(__file__).parent / "data"


def _geos_16_grid(dset: xr.Dataset) -> Any:
    """Calculates the AreaDefinition for GOES-16/ABI.

    Args:
        dset: The xarray Dataset containing the GOES-16 data.
              Must have 'goes_imager_projection' and 'x', 'y' coordinates.

    Returns:
        pyresample.geometry.AreaDefinition: The area definition for the grid.
    """
    from pyresample import geometry

    projection = dset.goes_imager_projection
    h = projection.perspective_point_height
    a = projection.semi_major_axis
    b = projection.semi_minor_axis
    lon_0 = projection.longitude_of_projection_origin
    sweep = projection.sweep_angle_axis
    x = dset.x * h
    y = dset.y * h
    x_ll = x[0]  # lower left corner
    x_ur = x[-1]  # upper right corner
    y_ll = y[0]  # lower left corner
    y_ur = y[-1]  # upper right corner
    x_h = (x_ur - x_ll) / (len(x) - 1.0) / 2.0  # 1/2 grid size
    y_h = (y_ur - y_ll) / (len(y) - 1.0) / 2.0  # 1/2 grid size
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

    area = geometry.AreaDefinition(
        "GEOS_ABI", "ABI", "GOES_ABI", proj_dict, len(x), len(y), np.asarray(area_extent)
    )
    return area


def _get_sinu_grid_df() -> pd.DataFrame:
    """Reads the sinusoidal grid boundary file.

    Returns:
        pd.DataFrame: DataFrame containing grid boundaries.
    """
    f = DATA_DIR / "sn_bound_10deg.txt"
    if not f.exists():
        raise FileNotFoundError(f"Sinusoidal grid file not found at {f}")

    td = pd.read_csv(f, skiprows=4, delim_whitespace=True)
    td = td.assign(ihiv="h" + td.ih.astype(str).str.zfill(2) + "v" + td.iv.astype(str).str.zfill(2))
    return td


def _sinu_grid_latlon_boundary(h: int, v: int) -> Tuple[float, float, float, float]:
    """Get lat/lon boundaries for sinusoidal grid tile.

    Args:
        h: Horizontal tile index.
        v: Vertical tile index.

    Returns:
        Tuple[float, float, float, float]: (lonmin, latmin, lonmax, latmax)
    """
    td = _get_sinu_grid_df()
    o = td.loc[(td.ih == int(h)) & (td.iv == int(v))]
    if o.empty:
        raise ValueError(f"Tile h={h}, v={v} not found in grid definition.")

    latmin = o.lat_min.iloc[0]
    lonmin = o.lon_min.iloc[0]
    latmax = o.lat_max.iloc[0]
    lonmax = o.lon_max.iloc[0]
    return lonmin, latmin, lonmax, latmax


def _get_sinu_xy(lon: Any, lat: Any) -> Tuple[Any, Any]:
    """Convert lon/lat to sinusoidal x/y.

    Args:
        lon: Longitude(s).
        lat: Latitude(s).

    Returns:
        Tuple: (x, y) coordinates.
    """
    from pyproj import Proj

    sinu = Proj("+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m")
    return sinu(lon, lat)


def _get_sinu_latlon(x: Any, y: Any) -> Tuple[Any, Any]:
    """Convert sinusoidal x/y to lon/lat.

    Args:
        x: X coordinate(s).
        y: Y coordinate(s).

    Returns:
        Tuple: (lon, lat) coordinates.
    """
    from pyproj import Proj

    # Create meshgrid if inputs are 1D arrays representing axes
    # But this function seems to expect x and y to be broadcastable or ready for meshgrid?
    # The original implementation did:
    # xv, yv = meshgrid(x, y)
    # So it expects 1D x and y axes.
    xv, yv = np.meshgrid(x, y)
    sinu = Proj(
        "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m, +R=6371007.181"
    )
    return sinu(xv, yv, inverse=True)


def get_sinu_area_extent(lonmin: float, latmin: float, lonmax: float, latmax: float) -> Tuple[float, float, float, float]:
    """Calculate area extent in sinusoidal projection.

    Args:
        lonmin: Minimum longitude.
        latmin: Minimum latitude.
        lonmax: Maximum longitude.
        latmax: Maximum latitude.

    Returns:
        Tuple: (xmin, ymin, xmax, ymax)
    """
    xmin, ymin = _get_sinu_xy(lonmin, latmin)
    xmax, ymax = _get_sinu_xy(lonmax, latmax)
    return (xmin, ymin, xmax, ymax)


def get_modis_latlon_from_swath_hv(h: int, v: int, dset: xr.Dataset) -> xr.Dataset:
    """Add longitude and latitude coordinates to MODIS dataset based on tile h/v.

    Args:
        h: Horizontal tile index.
        v: Vertical tile index.
        dset: Input dataset (must have 'x' and 'y' dimensions/coords).

    Returns:
        xr.Dataset: Dataset with 'longitude' and 'latitude' coordinates added.
    """
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
        "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 " "+b=6371007.181 +units=m"
    )
    return dset


def get_sinu_area_def(dset: xr.Dataset) -> Any:
    """Get pyresample AreaDefinition for sinusoidal grid from dataset.

    Args:
        dset: Input dataset with 'proj4_srs' and 'area_extent' attributes.

    Returns:
        pyresample.geometry.AreaDefinition: The area definition.
    """
    from pyproj import Proj
    from pyresample import utils

    p = Proj(dset.attrs["proj4_srs"])
    proj4_args = p.srs
    area_name = "MODIS Grid Def"
    area_id = "modis"
    proj_id = area_id
    area_extent = dset.attrs["area_extent"]
    nx, ny = dset.longitude.shape
    return utils.get_area_def(area_id, area_name, proj_id, proj4_args, nx, ny, area_extent)


def get_ioapi_pyresample_area_def(ds: Any, proj4_srs: str) -> Any:
    """Get pyresample AreaDefinition for IOAPI dataset.

    Args:
        ds: IOAPI dataset object (likely from monetio reader).
        proj4_srs: Proj4 string.

    Returns:
        pyresample.geometry.AreaDefinition: The area definition.
    """
    from pyresample import geometry, utils

    y_size = ds.NROWS
    x_size = ds.NCOLS
    projection = utils.proj4_str_to_dict(proj4_srs)
    proj_id = "IOAPI_Dataset"
    description = "IOAPI area_def for pyresample"
    area_id = "MONET_Object_Grid"
    x_ll, y_ll = ds.XORIG + ds.XCELL * 0.5, ds.YORIG + ds.YCELL * 0.5
    x_ur, y_ur = (
        ds.XORIG + (ds.NCOLS * ds.XCELL) + 0.5 * ds.XCELL,
        ds.YORIG + (ds.YCELL * ds.NROWS) + 0.5 * ds.YCELL,
    )
    area_extent = (x_ll, y_ll, x_ur, y_ur)
    area_def = geometry.AreaDefinition(
        area_id, description, proj_id, projection, x_size, y_size, area_extent
    )
    return area_def


def get_generic_projection_from_proj4(lat: Any, lon: Any, proj4_srs: str) -> Any:
    """Calculate optimal bounding box AreaDefinition from lat/lon and proj4 string.

    Args:
        lat: Latitude array.
        lon: Longitude array.
        proj4_srs: Proj4 string describing the projection.

    Returns:
        pyresample.geometry.AreaDefinition: The area definition.
    """
    try:
        from pyresample.geometry import SwathDefinition
        from pyresample.utils import proj4_str_to_dict
    except ImportError:
        print("please install pyresample to use this functionality")
        return None

    swath = SwathDefinition(lats=lat, lons=lon)
    area = swath.compute_optimal_bb_area(proj4_str_to_dict(proj4_srs))
    return area


def get_optimal_cartopy_proj(lat: Any, lon: Any, proj4_srs: str) -> Any:
    """Get optimal Cartopy projection.

    Args:
        lat: Latitude array.
        lon: Longitude array.
        proj4_srs: Proj4 string.

    Returns:
        cartopy.crs.Projection: The cartopy projection.
    """
    area = get_generic_projection_from_proj4(lat, lon, proj4_srs)
    if area is None:
        return None
    return area.to_cartopy_crs()


def _ioapi_grid_from_dataset(ds: Any, earth_radius: float = 6370000) -> str:
    """Get the IOAPI projection out of the file into proj4 string.

    Args:
        ds: Dataset with IOAPI metadata attributes (P_ALP, P_BET, etc.).
        earth_radius: Earth radius in meters.

    Returns:
        str: Proj4 string.
    """

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
        # Lambert
        p4 = (
            "+proj=lcc +lat_1={lat_1} +lat_2={lat_2} "
            "+lat_0={lat_0} +lon_0={lon_0} "
            "+x_0=0 +y_0=0 +datum=WGS84 +units=m +a={r} +b={r}"
        )
        p4 = p4.format(**pargs)
    elif proj_id == 4:
        # Polar stereo
        p4 = "+proj=stere +lat_ts={lat_1} +lon_0={lon_0} +lat_0=90.0" "+x_0=0 +y_0=0 +a={r} +b={r}"
        p4 = p4.format(**pargs)
    elif proj_id == 3:
        # Mercator
        p4 = (
            "+proj=merc +lat_ts={lat_1} " "+lon_0={center_lon} " "+x_0={x0} +y_0={y0} +a={r} +b={r}"
        )
        p4 = p4.format(**pargs)
    else:
        raise NotImplementedError("IOAPI proj not implemented yet: " "{}".format(proj_id))
    # area_def = _get_ioapi_pyresample_area_def(ds)
    return p4  # , area_def


def grid_from_dataset(ds: Any, earth_radius: float = 6370000) -> Optional[str]:
    """Identify and return grid projection from dataset.

    Args:
        ds: Input dataset.
        earth_radius: Earth radius in meters.

    Returns:
        str: Proj4 string or None if not found.
    """
    # maybe its an IOAPI file
    if hasattr(ds, "IOAPI_VERSION") or hasattr(ds, "P_ALP"):
        # IOAPI_VERSION
        return _ioapi_grid_from_dataset(ds, earth_radius=earth_radius)

    # Try out platte carree
    # return _lonlat_grid_from_dataset(ds)
    return None
