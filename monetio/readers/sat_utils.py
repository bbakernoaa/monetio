"""Satellite Reader Utilities"""

import datetime
from typing import Optional

import pandas as pd
import xarray as xr


def standardize_satellite_coords(
    ds: xr.Dataset,
    lat_name: str = "Latitude",
    lon_name: str = "Longitude",
    y_dim: str = "Rows",
    x_dim: str = "Columns",
) -> xr.Dataset:
    """
    Standardize satellite swath/gridded coordinates and dimensions.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    lat_name : str, optional
        Name of the latitude coordinate in the file, by default "Latitude".
    lon_name : str, optional
        Name of the longitude coordinate in the file, by default "Longitude".
    y_dim : str, optional
        Name of the y/row dimension in the file, by default "Rows".
    x_dim : str, optional
        Name of the x/column dimension in the file, by default "Columns".

    Returns
    -------
    xr.Dataset
        Dataset with standardized dimensions (y, x) and coordinates (latitude, longitude).
    """
    rename_dict = {}
    if y_dim in ds.dims:
        rename_dict[y_dim] = "y"
    if x_dim in ds.dims:
        rename_dict[x_dim] = "x"

    if rename_dict:
        ds = ds.rename(rename_dict)

    coord_rename = {}
    if lat_name in ds.variables and lat_name != "latitude":
        coord_rename[lat_name] = "latitude"
    if lon_name in ds.variables and lon_name != "longitude":
        coord_rename[lon_name] = "longitude"

    if coord_rename:
        ds = ds.rename(coord_rename)

    if "latitude" in ds.variables:
        ds["latitude"].attrs.update({"units": "degrees_north", "standard_name": "latitude"})
    if "longitude" in ds.variables:
        ds["longitude"].attrs.update({"units": "degrees_east", "standard_name": "longitude"})

    return ds


def add_time_coord(
    ds: xr.Dataset,
    time_val: Optional[datetime.datetime] = None,
    time_attr: Optional[str] = None,
) -> xr.Dataset:
    """
    Add a time dimension and coordinate to the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    time_val : datetime.datetime, optional
        Explicit time value to use.
    time_attr : str, optional
        Attribute name to extract time from if time_val is None.

    Returns
    -------
    xr.Dataset
        Dataset with 'time' dimension and coordinate.
    """
    if time_val is None and time_attr is not None:
        if time_attr in ds.attrs:
            try:
                time_val = pd.to_datetime(ds.attrs[time_attr])
            except (ValueError, TypeError):
                pass

    if time_val is not None:
        if isinstance(time_val, str):
            time_val = pd.to_datetime(time_val)
        ds = ds.expand_dims("time")
        ds["time"] = [time_val]

    return ds
