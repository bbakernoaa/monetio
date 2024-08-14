"""Read TROPOMI L2 NO2 data."""

import logging
import os
import sys
from collections import OrderedDict
from datetime import datetime
from glob import glob
from pathlib import Path

import numpy as np
import xarray as xr
from cftime import num2date
from netCDF4 import Dataset


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

    dso = Dataset(fname, "r")

    lon_var = dso.groups["PRODUCT"]["longitude"]
    lat_var = dso.groups["PRODUCT"]["latitude"]

    ref_time_var = dso.groups["PRODUCT"]["time"]
    ref_time_val = np.datetime64(num2date(ref_time_var[:].item(), ref_time_var.units))
    dtime_var = dso.groups["PRODUCT"]["delta_time"]
    dtime = xr.DataArray(dtime_var[:].squeeze(), dims=("y",)).astype("timedelta64[ms]")

    ds["lon"] = (
        ("y", "x"),
        lon_var[:].squeeze(),
        {"long_name": lon_var.long_name, "units": lon_var.units},
    )
    ds["lat"] = (
        ("y", "x"),
        lat_var[:].squeeze(),
        {"long_name": lat_var.long_name, "units": lat_var.units},
    )
    ds["time"] = ((), ref_time_val, {"long_name": "reference time"})
    ds["scan_time"] = ds["time"] + dtime
    ds["scan_time"].attrs.update({"long_name": "scan time"})
    ds = ds.set_coords(["lon", "lat", "time", "scan_time"])
    ds.attrs["reference_time_string"] = ref_time_val.astype(datetime).strftime(r"%Y%m%d")

    for varname in variable_dict:
        print(varname)
        values_var = dso.groups["PRODUCT"][varname]
        values = values_var[:].squeeze()

        fv = values_var.getncattr("_FillValue")
        if fv is not None:
            values[:][values[:] == fv] = np.nan

        if "fillvalue" in variable_dict[varname]:
            fillvalue = variable_dict[varname]["fillvalue"]
            values[:][values[:] == fillvalue] = np.nan

        if "scale" in variable_dict[varname]:
            values[:] = variable_dict[varname]["scale"] * values[:]

        if "minimum" in variable_dict[varname]:
            minimum = variable_dict[varname]["minimum"]
            values[:][values[:] < minimum] = np.nan

        if "maximum" in variable_dict[varname]:
            maximum = variable_dict[varname]["maximum"]
            values[:][values[:] > maximum] = np.nan

        ds[varname] = (
            ("y", "x"),
            values,
            {"long_name": values_var.long_name, "units": values_var.units},
        )

        if "quality_flag_min" in variable_dict[varname]:
            ds.attrs["quality_flag"] = varname
            ds.attrs["quality_thresh_min"] = variable_dict[varname]["quality_flag_min"]

    dso.close()

    return ds


def apply_quality_flag(ds):
    """Mask variables in place based on quality flag.

    Parameters
    ----------
    ds : xarray.Dataset
    """
    if "quality_flag" in ds.attrs:
        quality_flag = ds[ds.attrs["quality_flag"]]
        quality_thresh_min = ds.attrs["quality_thresh_min"]

        # Apply the quality thresh minimum to all variables in ds
        for varname in ds:
            print(varname)
            if varname != ds.attrs["quality_flag"]:
                logging.debug(varname)
                values = ds[varname].values
                values[quality_flag <= quality_thresh_min] = np.nan


def open_dataset(fnames, variable_dict, debug=False):
    """
    Parameters
    ----------
    fnames : str
        Glob expression for input file paths.
    variable_dict : dict
        Mapping of variable names to a dict of attributes
        that, if provided, will be used when processing the data
        (``fillvalue``, ``scale``, ``maximum``, ``minimum``, ``quality_flag_min``).
        A variable's attribute dict is allowed to be empty (``{}``).
        Or, instead, you can pass a single variable name as a string
        or a sequence of variable names.
    debug : bool
        Set logging level to debug.

    Returns
    -------
    OrderedDict
        Dict mapping reference time string (date) to :class:`xarray.Dataset` of the granule.
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
                print("environment variable not defined: " + subpath)
                exit(1)
            else:
                fnames = fnames.replace(subpath, envval)

    print(fnames)
    files = sorted(glob(fnames))
    granules = OrderedDict()
    for file in files:
        granule = _open_one_dataset(file, variable_dict)
        apply_quality_flag(granule)
        granules[granule.attrs["reference_time_string"]] = granule

    return granules
