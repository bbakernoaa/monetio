"""Read TROPOMI L2 NO2 data.

TROPOspheric Monitoring Instrument (TROPOMI) instrument.

http://www.tropomi.eu
https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-5p
"""

import logging
import os
import sys
import warnings
from collections import OrderedDict
from datetime import datetime
from glob import glob
from pathlib import Path

import h5netcdf
import numpy as np
import xarray as xr
from cftime import num2date


def _open_one_dataset(fname, variable_dict):
    """
    Parameters
    ----------
    fname : str
        Input file path.
    variable_dict : dict

    Returns
    -------
    xarray.Dataset
    """
    print("reading " + fname)

    ds = xr.Dataset()

    dso = h5netcdf.File(fname, "r")

    lon_var = dso["PRODUCT/longitude"]
    lat_var = dso["PRODUCT/latitude"]

    ref_time_var = dso["PRODUCT/time"]
    ref_time_val = np.datetime64(num2date(ref_time_var[:].item(), ref_time_var.attrs["units"]))
    dtime_var = dso["PRODUCT/delta_time"]
    dtime = xr.DataArray(dtime_var[:].squeeze(), dims=("y",)).astype("timedelta64[ms]")

    ds["lon"] = (
        ("y", "x"),
        lon_var[:].squeeze(),
        dict(lon_var.attrs),
    )
    ds["lat"] = (
        ("y", "x"),
        lat_var[:].squeeze(),
        dict(lat_var.attrs),
    )
    ds["time"] = ((), ref_time_val, {"long_name": "reference time"})
    ds["scan_time"] = ds["time"] + dtime
    ds["scan_time"].attrs.update({"long_name": "scan time"})
    ds = ds.set_coords(["lon", "lat", "time", "scan_time"])
    ds.attrs["reference_time_string"] = ref_time_val.astype(datetime).strftime(r"%Y-%m-%d")

    def _get_var(group_path, var_name):
        # Try direct access first using full path
        full_path = f"{group_path}/{var_name}".strip("/")
        if full_path in dso:
            return dso[full_path]

        # Fallback to recursive search
        def search(g):
            if var_name in g.variables:
                return g[var_name]
            for sg_name in g.groups:
                res = search(g.groups[sg_name])
                if res is not None:
                    return res
            return None

        res = search(dso)
        if res is not None:
            return res
        raise KeyError(f"Variable {var_name} not found in {fname}")

    def get_extra(varname_, *, dct_=None, default_group="PRODUCT"):
        """Get non-varname variables."""
        if dct_ is None:
            dct_ = variable_dict.get(varname_, {})
        group_name = dct_.get("group", default_group)
        if isinstance(group_name, list):
            group_name = group_name[0]
        return _get_values(_get_var(group_name, varname_), dct_)

    for varname, dct in variable_dict.items():
        print(f"- {varname}")
        if dct is None or (isinstance(dct, str) and dct in {"None", "NONE"}):
            dct = {}

        if varname == "preslev":
            # Compute layered pressure and tropopause pressure

            itrop_ = get_extra("tm5_tropopause_layer_index", dct_=dct)
            itrop_[itrop_ == 0] = 1  # Avoid trop in surface layer
            itrop = xr.DataArray(itrop_, dims=("y", "x"))
            a = xr.DataArray(data=get_extra("tm5_constant_a", dct_=dct), dims=("z", "v"))
            b = xr.DataArray(data=get_extra("tm5_constant_b", dct_=dct), dims=("z", "v"))
            psfc = xr.DataArray(
                data=get_extra(
                    "surface_pressure",
                    default_group="PRODUCT/SUPPORT_DATA/INPUT_DATA",
                    dct_=dct,
                ),
                dims=("y", "x"),
            )

            # Mid-layer pressure
            assert a.sizes["v"] == 2, "base and top"
            p = ((a.isel(v=0) + b.isel(v=0) * psfc) + (a.isel(v=1) + b.isel(v=1) * psfc)) / 2
            ds["preslev"] = p
            ds["preslev"].attrs.update({"long_name": "mid-layer pressure", "units": "Pa"})

            # Tropopause pressure
            ptrop = xr.full_like(itrop, np.nan, dtype=ds["preslev"].dtype)
            for i in np.unique(itrop):
                if np.isnan(i) or i < 0 or i >= p.sizes["z"]:
                    continue
                ptrop = xr.where(itrop == i, p.isel(z=int(i)), ptrop)
            ds["troppres"] = ptrop
            ds["troppres"].attrs.update({"long_name": "tropopause pressure", "units": "Pa"})

        elif varname in {"latitude_bounds", "longitude_bounds"}:
            group_name = dct.get("group", "PRODUCT/SUPPORT_DATA/GEOLOCATIONS")
            values = _get_values(_get_var(group_name, varname), dct)
            assert values.shape[-1] == 4
            for i in range(4):
                ds[f"{varname}_{i}"] = (
                    ("y", "x"),
                    values[:, :, i],
                    {"long_name": f"{varname} {i}"},
                )

        else:
            group_name = dct.get("group", "PRODUCT")
            var = _get_var(group_name, varname)
            values = _get_values(var, dct)

            if values.ndim == 2:
                dims = ("y", "x")
            elif values.ndim == 3:
                dims = ("y", "x", "z")
            else:
                raise ValueError(f"unexpected ndim ({varname}): {values.ndim}")

            ds[varname] = (
                dims,
                values,
                dict(var.attrs),
            )

            if "quality_flag_min" in dct:
                ds.attrs["quality_flag"] = varname
                ds.attrs["quality_thresh_min"] = dct["quality_flag_min"]
                ds.attrs["var_applied"] = dct.get("var_applied", [])

    dso.close()

    return ds


def _get_values(var, dct):
    """Take Variable, squeeze, tweak values based on user-provided attribute dict,
    and return NumPy array."""

    values = var[:].squeeze()

    # manual masking
    fill_value = var.attrs.get("_FillValue")
    if fill_value is None:
        fill_value = var.attrs.get("missing_value")
    if fill_value is not None:
        values = np.ma.masked_values(values, fill_value, rtol=1e-5, copy=False)

    # fallback for very large values (typical of unmasked NetCDF)
    if np.issubdtype(values.dtype, np.floating):
        values = np.ma.masked_greater(values, 1e36, copy=False)

    # manual scaling
    scale = var.attrs.get("scale_factor", 1.0)
    offset = var.attrs.get("add_offset", 0.0)
    if scale != 1.0 or offset != 0.0:
        values = values * scale + offset

    if np.ma.is_masked(values):
        logging.info(f"{var.name} already masked")

    scale = dct.get("scale")
    if scale is not None:
        values *= float(scale)

    # Note netcdf4-python masks based on nc _FillValue attr automatically
    fv = dct.get("fillvalue")
    if fv is not None:
        values = np.ma.masked_values(values, fv, atol=0, copy=False)

    minimum = dct.get("minimum")
    if minimum is not None:
        values = np.ma.masked_less(values, minimum, copy=False)

    maximum = dct.get("maximum")
    if maximum is not None:
        values = np.ma.masked_greater(values, maximum, copy=False)

    return values


def apply_quality_flag(ds):
    """Mask variables in place based on quality flag.

    Parameters
    ----------
    ds : xarray.Dataset
    """
    if "quality_flag" in ds.attrs:
        quality_flag = ds[ds.attrs["quality_flag"]]  # quality flag variable (float)
        quality_thresh_min = ds.attrs["quality_thresh_min"]

        # Apply the quality thresh minimum to selected variables
        for varname in ds.attrs["var_applied"]:
            logging.debug(f"applying quality flag to {varname}")
            values = ds[varname].values
            values[quality_flag <= quality_thresh_min] = np.nan


def open_dataset(fnames, variable_dict, debug=False):
    """Open one or more TROPOMI L2 NO2 files.

    Parameters
    ----------
    fnames : str
        Glob expression for input file paths.
    variable_dict : dict
        Mapping of variable names to a dict of attributes
        that, if provided, will be used when processing the data
        (``fillvalue``, ``scale``, ``maximum``, ``minimum``, ``quality_flag_min``).
        A variable's attribute dict is allowed to be empty (``{}``).
        For ``longitude_bounds`` and ``latitude_bounds``,
        variables ``longitude_bounds_{1..4}`` and/or ``latitude_bounds_{1..4}`` are created.
        For ``preslev``, you can specify attributes for the variables used to calculate pressure
        (``tm5_tropopause_layer_index``, ``tm5_constant_a``, ``tm5_constant_b``, ``surface_pressure``),
        and variables ``preslev`` (mid-layer pressure)
        and ``troppres`` (tropopause pressure) are created.
        For any variable,
        a non-default group name can be specified with the key ``group`` (use ``/`` for nesting).
        Or, instead, you can pass a single variable name as a string
        or a sequence of variable names.
    debug : bool
        Set logging level to debug.

    Returns
    -------
    OrderedDict
        Dict mapping reference time string (date, YYYY-MM-DD)
        to a list of :class:`xarray.Dataset` granules.
    """
    if debug:
        logging_level = logging.DEBUG
        logging.basicConfig(stream=sys.stdout, level=logging_level)

    if isinstance(fnames, Path):
        fnames = fnames.as_posix()

    if isinstance(variable_dict, str):
        variable_dict = {variable_dict: {}}
    elif isinstance(variable_dict, dict):
        pass
    else:  # Assume sequence
        variable_dict = {varname: {} for varname in variable_dict}

    for subpath in fnames.split("/"):
        if "$" in subpath:
            envvar = subpath.replace("$", "")
            envval = os.getenv(envvar)
            if envval is None:
                raise RuntimeError(f"environment variable {envvar!r} not defined: " + subpath)
            else:
                fnames = fnames.replace(subpath, envval)

    files = sorted(glob(fnames))
    granules = OrderedDict()
    for file in files:
        granule = _open_one_dataset(file, variable_dict)
        apply_quality_flag(granule)
        key = granule.attrs["reference_time_string"]
        if key in granules:
            granules[key].append(granule)
        else:
            granules[key] = [granule]

    return granules


def read_trpdataset(*args, **kwargs):
    """Alias for :func:`open_dataset`."""
    warnings.warn(
        "read_trpdataset is an alias for open_dataset and may be removed in the future",
        FutureWarning,
        stacklevel=2,
    )
    return open_dataset(*args, **kwargs)
