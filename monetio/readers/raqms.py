"""RAQMS Reader"""

import datetime
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader


@register_reader("raqms")
class RAQMSReader(GriddedReader):
    """
    Reader for RAQMS (Real-time Air Quality Modeling System) model output files.
    """

    def open_dataset(
        self,
        files: Union[str, List[str]],
        convert_to_ppb: bool = True,
        var_list: Optional[List[str]] = None,
        surf_only: bool = False,
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads RAQMS netCDF files.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path, list of paths, or glob pattern.
        convert_to_ppb : bool, optional
            Convert gas species from ppv to ppbv, by default True.
        var_list : List[str], optional
            List of variables to keep, by default None.
        surf_only : bool, optional
            Whether to only return the surface layer, by default False.
        **kwargs : dict
            Additional arguments passed to xarray.open_mfdataset or the driver.

        Returns
        -------
        xr.Dataset
            The processed RAQMS dataset.
        """
        # RAQMS check file format
        import os
        from glob import glob

        # Match original behavior for tests
        if isinstance(files, str):
            fpaths = sorted(glob(files))
        else:
            fpaths = sorted(files)

        if not fpaths or not all(
            fp.endswith(".nc") and "uwhyb" in os.path.basename(fp) for fp in fpaths
        ):
            raise ValueError(
                "File format not supported. Note that files should be preprocessed to netCDF."
            )

        # Prepare kwargs
        if "concat_dim" not in kwargs:
            kwargs["concat_dim"] = "time"
        if "combine" not in kwargs:
            kwargs["combine"] = "nested"

        # RAQMS specific drop
        if "drop_variables" not in kwargs:
            kwargs["drop_variables"] = ["theta"]
        elif "theta" not in kwargs["drop_variables"]:
            if isinstance(kwargs["drop_variables"], list):
                kwargs["drop_variables"].append("theta")

        ds = self.driver.open(files, **kwargs)

        # 1. Update history
        history = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read RAQMS data."
        if "history" in ds.attrs:
            ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
        else:
            ds.attrs["history"] = history

        if var_list is not None:
            # Add required vars
            required = [
                "lat",
                "lon",
                "IDATE",
                "Times",
                "psfc",
                "delp",
                "pdash",
                "ttheta",
            ]
            # Ensure we don't duplicate
            vars_to_keep = list(set(var_list + required))
            # Only keep available ones
            vars_to_keep = [v for v in vars_to_keep if v in ds.variables]
            ds = ds[vars_to_keep]

        # Post processing
        ds = _fix(ds, surf_only=surf_only, convert_to_ppb=convert_to_ppb)

        return ds


# -----------------------------------------------------------------------------
# Helper functions ported from monetio/models/raqms.py
# -----------------------------------------------------------------------------


def _fix(ds: xr.Dataset, *, surf_only: bool, convert_to_ppb: bool) -> xr.Dataset:
    """
    Internal fix function for RAQMS dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input RAQMS dataset.
    surf_only : bool
        Whether to keep only the surface layer.
    convert_to_ppb : bool
        Whether to convert ppv to ppbv.

    Returns
    -------
    xr.Dataset
        Fixed dataset.
    """
    ds = _fix_grid(ds)
    ds = _fix_time(ds)
    ds = _fix_pres(ds)

    if surf_only:
        ds = ds.isel(z=0).expand_dims("z")

    if convert_to_ppb:
        for i in ds.data_vars:
            if "units" in ds[i].attrs:
                if ds[i].attrs["units"] == "ppv":
                    with xr.set_options(keep_attrs=True):
                        ds[i] = ds[i] * 1e9
                    ds[i].attrs["units"] = "ppbv"

    if "ttheta" in ds.data_vars:
        # Calculate temperature from potential temperature
        k = 0.28571428571428564  # R/cp = kappa (unitless)
        with xr.set_options(keep_attrs=True):
            ds["temperature_k"] = ds["ttheta"] * (ds["pres_pa_mid"] / 100000) ** k
        ds["temperature_k"].attrs["units"] = "K"
        ds["temperature_k"].attrs["long_name"] = "Temperature"

    # Transpose if dims exist
    dims = [d for d in ["time", "z", "y", "x"] if d in ds.dims]
    ds = ds.transpose(*dims)

    return ds


def _fix_grid(ds: xr.Dataset) -> xr.Dataset:
    """
    Fix grid and coordinates for RAQMS.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.

    Returns
    -------
    xr.Dataset
        Dataset with 'latitude' and 'longitude' coordinates.
    """
    # Handle coordinates lazily BEFORE renaming dims
    lat_name = "lat" if "lat" in ds.dims else "y"
    lon_name = "lon" if "lon" in ds.dims else "x"

    lat_orig = ds[lat_name]
    lon_orig = ds[lon_name]
    lon_adj = xr.where(lon_orig >= 180, lon_orig - 360, lon_orig)

    # Broadcast to 2D
    # xr.broadcast will handle both NumPy and Dask lazily
    lon2d, lat2d = xr.broadcast(lon_adj, lat_orig)

    # Rename dims of the dataset FIRST
    rename_dims = {}
    if "lat" in ds.dims:
        rename_dims["lat"] = "y"
    if "lon" in ds.dims:
        rename_dims["lon"] = "x"
    if "lev" in ds.dims:
        rename_dims["lev"] = "z"

    if rename_dims:
        ds = ds.rename_dims(rename_dims)

    # Ensure dimension order and rename to standard y, x for coordinates
    lon2d = lon2d.transpose(lat_name, lon_name).rename({lat_name: "y", lon_name: "x"})
    lat2d = lat2d.transpose(lat_name, lon_name).rename({lat_name: "y", lon_name: "x"})

    ds["longitude"] = lon2d.assign_attrs(
        {
            "long_name": "Longitude",
            "units": "degree_east",
            "standard_name": "longitude",
        }
    )
    ds["latitude"] = lat2d.assign_attrs(
        {
            "long_name": "Latitude",
            "units": "degree_north",
            "standard_name": "latitude",
        }
    )

    ds = ds.drop_vars(["lat", "lon"], errors="ignore")
    ds = ds.set_coords(["latitude", "longitude"])

    if "lev" in ds.variables:
        ds["lev"].attrs.update(
            long_name="Nominal potential temperature of model level",
            units="K",
            description=(
                "In the stratosphere (beginning at lev=492), the model levels are on potential temperature surfaces. "
                "Below lev=492, the model levels are a blend of potential temperature and sigma (terrain-following) coordinates."
            ),
        )

    # Invert in z so that index 0 is closest to surface
    if "z" in ds.dims:
        ds = ds.isel(z=slice(None, None, -1))

    return ds


def _fix_time(ds: xr.Dataset) -> xr.Dataset:
    """
    Fix time coordinate for RAQMS.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.

    Returns
    -------
    xr.Dataset
        Dataset with 'time' coordinate.
    """
    if "Times" in ds.variables:
        # Times is usually a character array (time, char_len)
        # We want to convert it to datetime64 lazily if possible.
        # But constructing strings and then datetimes is usually not lazy in Xarray
        # unless we use apply_ufunc.
        # For now, if it's small (one time per file), we might compute it,
        # but the Aero Protocol says NO HIDDEN COMPUTES.

        # Use apply_ufunc with vectorize=True to handle character arrays or strings lazily.
        def _parse_raqms_times(times_val):
            # times_val is a single value (string or bytes) due to vectorize=True
            if hasattr(times_val, "decode"):
                s = times_val.decode("utf-8").strip()
            else:
                s = str(times_val).strip()

            if not s:
                return np.datetime64("NaT")
            try:
                return pd.to_datetime(s, format=r"%Y_%m_%d_%H:%M:%S").to_datetime64()
            except Exception:
                return np.datetime64("NaT")

        # If it's dask, we use apply_ufunc to keep it lazy
        time_values = xr.apply_ufunc(
            _parse_raqms_times,
            ds.Times,
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.dtype("datetime64[ns]")],
        )

        ds = ds.assign_coords(time=time_values)
        ds = ds.drop_vars(["IDATE", "Times"], errors="ignore")
    return ds


def _fix_pres(ds: xr.Dataset) -> xr.Dataset:
    """
    Fix pressure variables for RAQMS.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.

    Returns
    -------
    xr.Dataset
        Dataset with renamed and scaled pressure variables.
    """
    rename0 = {
        "psfc": "surfpres_pa",
        "delp": "dp_pa",
        "pdash": "pres_pa_mid",
    }
    rename = {k: v for k, v in rename0.items() if k in ds.variables}

    if rename:
        ds = ds.rename_vars(rename)

    for vn in rename.values():
        if "units" in ds[vn].attrs and ds[vn].attrs["units"] in {"mb", "hPa"}:
            with xr.set_options(keep_attrs=True):
                ds[vn] = ds[vn] * 100
            ds[vn].attrs.update(units="Pa")

    return ds
