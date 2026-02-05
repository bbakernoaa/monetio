"""TEMPO L2 NO2 data file reader.

History:


"""

import logging
import os
import sys
import warnings
from glob import glob
from pathlib import Path

import numpy as np
import xarray as xr
from cftime import num2pydate
from netCDF4 import Dataset


def _open_one_dataset(fname, variable_dict):
    """Read locally stored INTELSAT TEMPO NO2 level 2

    Parameters
    ----------
    fname : str
        Local path to netCDF4 (HDF5) file.
    variable_dict : dict

    Returns
    -------
    ds : xr.Dataset
    """
    print("reading " + fname)

    ds = xr.Dataset()

    # Cache for opened groups
    _groups = {}

    def get_group(group_name):
        if group_name not in _groups:
            _groups[group_name] = xr.open_dataset(
                fname, group=group_name, engine="h5netcdf", decode_times=False
            )
        return _groups[group_name]

    ds_geo = get_group("geolocation")
    lon_var = ds_geo["longitude"]
    lat_var = ds_geo["latitude"]
    time_var = ds_geo["time"]
    time_units = time_var.attrs.get("units", "")

    ds["lon"] = (
        ("x", "y"),
        lon_var.values.squeeze(),
        {"long_name": lon_var.attrs.get("long_name"), "units": lon_var.attrs.get("units")},
    )
    ds["lat"] = (
        ("x", "y"),
        lat_var.values.squeeze(),
        {"long_name": lat_var.attrs.get("long_name"), "units": lat_var.attrs.get("units")},
    )
    ds["time"] = (
        ("time",),
        num2pydate(time_var.values.squeeze(), time_units),
        {"long_name": time_var.attrs.get("long_name")},
    )
    ds = ds.set_coords(["time", "lon", "lat"])

    # Root attributes
    with xr.open_dataset(fname, engine="h5netcdf") as ds_root:
        ds.attrs["reference_time_string"] = ds_root.attrs.get("time_coverage_start")
        ds.attrs["granule_number"] = ds_root.attrs.get("granule_num")
        ds.attrs["scan_num"] = ds_root.attrs.get("scan_num")

    if ("pressure" in variable_dict) and "surface_pressure" not in variable_dict:
        warnings.warn(
            "Calculating pressure in TEMPO data requires surface_pressure. "
            + "Adding surface_pressure to output variables"
        )
        variable_dict["surface_pressure"] = {}

    for varname in variable_dict:
        if varname in [
            "main_data_quality_flag",
            "vertical_column_troposphere",
            "vertical_column_stratosphere",
            "vertical_column_troposphere_uncertainty",
            "vertical_column",
            "vertical_column_uncertainty",
        ]:
            group_name = "product"
        elif varname in [
            "latitude_bounds",
            "longitude_bounds",
            "solar_zenith_angle",
            "solar_azimuth_angle",
            "viewing_zenith_angle",
            "viewing_azimuth_angle",
            "relative_azimuth_angle",
        ]:
            group_name = "geolocation"
        elif varname in [
            "vertical_column_total",
            "vertical_column_total_uncertainty",
            "fitted_slant_column",
            "fitted_slant_column_uncertainty",
            "snow_ice_fraction",
            "terrain_height",
            "ground_pixel_quality_flag",
            "surface_pressure",
            "tropopause_pressure",
            "scattering_weights",
            "gas_profile",
            "albedo",
            "temperature_profile",
            "amf",
            "amf_total",
            "amf_diagnostic_flag",
            "eff_cloud_fraction",
            "amf_cloud_fraction",
            "amf_cloud_pressure",
            "amf_troposphere",
            "amf_stratosphere",
            "background_correction",
        ]:
            group_name = "support_data"
        elif varname in ["fit_rms_residual", "fit_convergence_flag"]:
            group_name = "qa_statistics"

        values_var = get_group(group_name)[varname]
        values = values_var.values.copy()
        if not np.ma.is_masked(values):
            if "_FillValue" in values_var.attrs:
                values = np.ma.masked_equal(values, values_var.attrs["_FillValue"])
            elif "missing_value" in values_var.attrs:
                values = np.ma.masked_equal(values, values_var.attrs["missing_value"])
        values = values.squeeze()

        if "scale" in variable_dict[varname]:
            values[:] = variable_dict[varname]["scale"] * values[:]

        if "minimum" in variable_dict[varname]:
            minimum = variable_dict[varname]["minimum"]
            values = np.ma.masked_less(values, minimum, copy=False)

        if "maximum" in variable_dict[varname]:
            maximum = variable_dict[varname]["maximum"]
            values = np.ma.masked_greater(values, maximum, copy=False)

        if "corner" in values_var.dims:
            ds[varname] = (("x", "y", "corner"), values, values_var.attrs)
        elif "swt_level" in values_var.dims:
            ds[varname] = (("x", "y", "swt_level"), values, values_var.attrs)
        else:
            ds[varname] = (("x", "y"), values, values_var.attrs)

        if "quality_flag_max" in variable_dict[varname]:
            ds.attrs["quality_flag"] = varname
            ds.attrs["quality_thresh_max"] = variable_dict[varname]["quality_flag_max"]

    for g in _groups.values():
        g.close()

    if "surface_pressure" in variable_dict:
        if ds["surface_pressure"].attrs["units"] == "hPa":
            HPA2PA = 100
            ds["surface_pressure"][:] = ds["surface_pressure"].values * HPA2PA
            ds["surface_pressure"].attrs["units"] = "Pa"
            ds["surface_pressure"].attrs["valid_min"] *= HPA2PA
            ds["surface_pressure"].attrs["valid_max"] *= HPA2PA
            ds["surface_pressure"].attrs["Eta_A"] *= HPA2PA
    if "pressure" in variable_dict:
        ds["pressure"] = calculate_pressure(ds)

    return ds


def calculate_pressure(ds):
    """Calculates pressure at layer and delta_pressure of layer

    Parameters
    ----------
    ds : xr.Dataset

    Returns
    -------
    layer_pressure : xr.DataArray
        Pressure at layer in Pa
    delta_pressure : xr.DataArray
        Difference of pressure in layer
    """
    surf_pressure = ds["surface_pressure"]
    eta_a = surf_pressure.Eta_A
    eta_b = surf_pressure.Eta_B
    n_layers = len(surf_pressure.Eta_A)
    press = np.zeros((n_layers, surf_pressure.shape[0], surf_pressure.shape[1]))
    for k in range(0, n_layers):
        press[k, :, :] = eta_a[k] + eta_b[k] * surf_pressure.values
    pressure = xr.DataArray(
        data=press,
        dims=("swt_level_stagg", "x", "y"),
        coords={
            "lon": (["x", "y"], surf_pressure.lon.values),
            "lat": (["x", "y"], surf_pressure.lat.values),
        },
        attrs={
            "long_name": "pressure",
            "units": surf_pressure.attrs["units"],
            "valid_min": surf_pressure.attrs["valid_min"],
            "valid_max": surf_pressure.attrs["valid_max"],
            "algorithm": "Calculated from hybrid coords as Eta_A + Eta_B * surface_pressure",
            "_FillValue": np.nan,
        },
    )
    return pressure


def apply_quality_flag(ds):
    """Mask variables in place based on quality flag

    Parameters
    ----------
    ds : xr.Dataset
    """
    if "quality_flag" in ds.attrs:
        quality_flag = ds[ds.attrs["quality_flag"]]
        quality_thresh_max = ds.attrs["quality_thresh_max"]

        # Apply the quality thersh maximum to all variables in ds
        for varname in ds:
            if varname != ds.attrs["quality_flag"]:
                logging.debug(varname)
                ds[varname] = ds[varname].where(~(quality_flag > quality_thresh_max))


def open_dataset(fnames, variable_dict, debug=False):
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
                raise Exception("Environment variable not defined: " + subpath)
            else:
                fnames = fnames.replace(subpath, envval)

    print(fnames)
    files = sorted(glob(fnames))

    granules = {}

    for file in files:
        granule = _open_one_dataset(file, variable_dict)
        apply_quality_flag(granule)
        granules[granule.attrs["reference_time_string"]] = granule

    return granules
