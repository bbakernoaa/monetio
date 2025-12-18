"""RAQMS Reader"""

import pandas as pd
import xarray as xr
from numpy import meshgrid

from .base import GriddedReader, register_reader


@register_reader("raqms")
class RAQMSReader(GriddedReader):
    def open_dataset(self, files, convert_to_ppb=True, var_list=None, surf_only=False, **kwargs):
        """
        Reads RAQMS netCDF files.
        """
        # RAQMS check file format
        # Check if uwhyb in name?
        # Let's rely on user passing correct files or glob.

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

        if var_list is not None:
            # Add required vars
            required = ["lat", "lon", "IDATE", "Times", "psfc", "delp", "pdash", "ttheta"]
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


def _fix(ds, *, surf_only, convert_to_ppb):
    ds = _fix_grid(ds)
    ds = _fix_time(ds)
    ds = _fix_pres(ds)

    if surf_only:
        ds = ds.isel(z=0).expand_dims("z")

    if convert_to_ppb:
        for i in ds.variables:
            if "units" in ds[i].attrs:
                if ds[i].attrs["units"] == "ppv":
                    with xr.set_options(keep_attrs=True):
                        ds[i] = ds[i] * 1e9
                    ds[i].attrs["units"] = "ppbv"

    if "ttheta" in ds.keys():
        # Calculate temperature from potential temperature
        k = 0.28571428571428564  # R/cp = kappa (unitless; value for dry air from metpy.constants)
        ds["temperature_k"] = ds["ttheta"] * (ds["pres_pa_mid"] / 100000) ** k
        ds["temperature_k"].attrs["units"] = "K"

    # Transpose if dims exist
    # Check dims existence
    dims = [d for d in ["time", "z", "y", "x"] if d in ds.dims]
    ds = ds.transpose(*dims)

    return ds


def _fix_grid(ds):
    lat = ds.lat.values
    lon = ds.lon.values
    lon[(lon >= 180)] -= 360
    lon, lat = meshgrid(lon, lat)

    # Rename dims
    rename_dims = {}
    if "lat" in ds.dims:
        rename_dims["lat"] = "y"
    if "lon" in ds.dims:
        rename_dims["lon"] = "x"
    if "lev" in ds.dims:
        rename_dims["lev"] = "z"

    ds = ds.rename_dims(rename_dims)
    ds = ds.drop_vars(["lat", "lon"], errors="ignore")

    ds["longitude"] = (
        ("y", "x"),
        lon,
        {
            "long_name": "Longitude",
            "units": "degree_east",
            "standard_name": "longitude",
        },
    )
    ds["latitude"] = (
        ("y", "x"),
        lat,
        {
            "long_name": "Latitude",
            "units": "degree_north",
            "standard_name": "latitude",
        },
    )
    ds = ds.reset_coords().set_coords(["latitude", "longitude"])

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


def _fix_time(ds):
    if "Times" in ds.variables:
        dtstr = ds.Times.values.astype(str)
        time = pd.to_datetime(dtstr, format=r"%Y_%m_%d_%H:%M:%S")
        if "time" not in ds.coords:
            # If 'time' dim exists but no coord, or if we need to replace it
            pass
        ds = ds.assign_coords(time=time)
        ds = ds.drop_vars(["IDATE", "Times"], errors="ignore")
    return ds


def _fix_pres(ds):
    rename0 = {
        "psfc": "surfpres_pa",
        "delp": "dp_pa",
        "pdash": "pres_pa_mid",
    }
    rename = {k: v for k, v in rename0.items() if k in ds.variables}

    ds = ds.rename_vars(rename)
    for vn in rename.values():
        if "units" in ds[vn].attrs and ds[vn].attrs["units"] in {"mb", "hPa"}:
            with xr.set_options(keep_attrs=True):
                ds[vn] *= 100
            ds[vn].attrs.update(units="Pa")

    return ds
