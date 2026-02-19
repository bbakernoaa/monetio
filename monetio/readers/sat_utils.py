"""Satellite Reader Utilities"""

import datetime
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr


def lazy_index_along_axis(data: xr.DataArray, index: xr.DataArray, dim: str) -> xr.DataArray:
    """
    Index a dimension using another DataArray lazily, handling both Eager and Dask.
    Fixes the 'vindex does not support indexing with dask objects' limitation.

    Parameters
    ----------
    data : xr.DataArray
        DataArray to index. Must have dimension `dim`.
    index : xr.DataArray
        DataArray of indices.
    dim : str
        Dimension name to index along.

    Returns
    -------
    xr.DataArray
        The indexed DataArray.
    """

    def _index_func(arr, idx):
        # In apply_ufunc with input_core_dims=[[dim], []], the core dimension
        # is moved to the last axis of arr.
        # arr shape: (..., dim_size)
        # idx shape: (...)
        idx_expanded = idx[..., np.newaxis]
        return np.take_along_axis(arr, idx_expanded, axis=-1).squeeze(axis=-1)

    return xr.apply_ufunc(
        _index_func,
        data,
        index,
        input_core_dims=[[dim], []],
        output_core_dims=[[]],
        dask="parallelized",
        output_dtypes=[data.dtype],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )


def apply_lazy_conversion(
    data: xr.DataArray, func: Callable, output_dtype: Union[str, np.dtype, type]
) -> xr.DataArray:
    """
    Apply a conversion function lazily to a DataArray using Aero Protocol.

    Parameters
    ----------
    data : xr.DataArray
        Input DataArray.
    func : Callable
        Function to apply. Should work on NumPy arrays.
    output_dtype : Union[str, np.dtype, type]
        Expected output dtype.

    Returns
    -------
    xr.DataArray
        Converted DataArray.
    """

    def _wrapped_func(x):
        res = func(x)
        # Ensure result is a NumPy array to avoid issues with Dask/Xarray
        # (e.g. DatetimeIndex causing transpose errors)
        if hasattr(res, "to_numpy"):
            return res.to_numpy()
        return np.asarray(res)

    return xr.apply_ufunc(
        _wrapped_func,
        data,
        dask="parallelized",
        output_dtypes=[output_dtype],
    )


def standardize_satellite_coords(
    ds: xr.Dataset,
    lat_name: str = "Latitude",
    lon_name: str = "Longitude",
    y_dim: Union[str, List[str]] = ["Rows", "scanline", "nlat", "lat", "nscan"],
    x_dim: Union[str, List[str]] = ["Columns", "ground_pixel", "nlon", "lon", "nstep"],
    z_dim: Union[str, List[str]] = ["Levels", "layer", "level"],
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
    y_dim : str or list of str, optional
        Name(s) of the y/row dimension in the file, by default ["Rows", "scanline"].
    x_dim : str or list of str, optional
        Name(s) of the x/column dimension in the file, by default ["Columns", "ground_pixel"].

    Returns
    -------
    xr.Dataset
        Dataset with standardized dimensions (y, x) and coordinates (latitude, longitude).
    """
    if isinstance(y_dim, str):
        y_dim = [y_dim]
    if isinstance(x_dim, str):
        x_dim = [x_dim]

    rename_dict = {}
    for y in y_dim:
        if y in ds.dims:
            rename_dict[y] = "y"
            break
    for x in x_dim:
        if x in ds.dims:
            rename_dict[x] = "x"
            break
    for z in z_dim:
        if z in ds.dims:
            rename_dict[z] = "z"
            break

    if rename_dict:
        ds = ds.rename(rename_dict)

    coord_rename = {}
    # Case insensitive search for lat/lon if not found exactly
    actual_lat = None
    if lat_name in ds.variables:
        actual_lat = lat_name
    else:
        for v in ds.variables:
            if v.lower() == lat_name.lower():
                actual_lat = v
                break

    actual_lon = None
    if lon_name in ds.variables:
        actual_lon = lon_name
    else:
        for v in ds.variables:
            if v.lower() == lon_name.lower():
                actual_lon = v
                break

    if actual_lat and actual_lat != "latitude":
        coord_rename[actual_lat] = "latitude"
    if actual_lon and actual_lon != "longitude":
        coord_rename[actual_lon] = "longitude"

    if coord_rename:
        ds = ds.rename(coord_rename)

    if "latitude" in ds.variables:
        ds["latitude"].attrs.update({"units": "degrees_north", "standard_name": "latitude"})
    if "longitude" in ds.variables:
        ds["longitude"].attrs.update({"units": "degrees_east", "standard_name": "longitude"})

    return ds


def update_history(ds: xr.Dataset, message: str) -> xr.Dataset:
    """
    Update the 'history' attribute of a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    message : str
        Message to add to history.

    Returns
    -------
    xr.Dataset
        Dataset with updated history.
    """
    history = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {message}"
    if "history" in ds.attrs:
        ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
    else:
        ds.attrs["history"] = history
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
