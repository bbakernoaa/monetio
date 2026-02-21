"""WRF-Chem Reader"""

import datetime
from functools import partial
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader


@register_reader("wrfchem")
class WRFChemReader(GriddedReader):
    """
    Reader for WRF-Chem and RAP-Chem model output files.
    """

    def open_dataset(
        self,
        files: Union[str, List[str]],
        convert_to_ppb: bool = True,
        mech: str = "racm_esrl_vcp",
        var_list: Optional[List[str]] = None,
        surf_only: bool = False,
        surf_only_nc: bool = False,
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads WRF-Chem netCDF files.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path, list of paths, or glob pattern.
        convert_to_ppb : bool, optional
            Convert gas species from ppmV to ppbV, by default True.
        mech : str, optional
            Mechanism for calculating sums, by default "racm_esrl_vcp".
        var_list : List[str], optional
            List of variables to include, by default None.
        surf_only : bool, optional
            Whether to only keep surface data, by default False.
        surf_only_nc : bool, optional
            Whether input data already contains only surface data, by default False.
        **kwargs : dict
            Additional arguments passed to the driver.

        Returns
        -------
        xr.Dataset
            The processed WRF-Chem dataset.
        """
        if "preprocess" not in kwargs:
            kwargs["preprocess"] = partial(
                wrfchem_preprocess,
                convert_to_ppb=convert_to_ppb,
                mech=mech,
                var_list=var_list,
                surf_only=surf_only,
                surf_only_nc=surf_only_nc,
            )

        if "combine" not in kwargs:
            kwargs["combine"] = "nested"
        if "concat_dim" not in kwargs:
            kwargs["concat_dim"] = "time"

        ds = self.driver.open(files, **kwargs)

        ds = self.harmonize(ds)

        # Update history
        history = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read WRF-Chem data."
        if "history" in ds.attrs:
            ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
        else:
            ds.attrs["history"] = history

        return ds


def wrfchem_preprocess(
    ds: xr.Dataset,
    *,
    convert_to_ppb: bool = True,
    mech: str = "racm_esrl_vcp",
    var_list: Optional[List[str]] = None,
    surf_only: bool = False,
    surf_only_nc: bool = False,
) -> xr.Dataset:
    """
    Preprocess function for a single WRF-Chem file following Aero Protocol.

    Parameters
    ----------
    ds : xr.Dataset
        Input WRF-Chem dataset.
    convert_to_ppb : bool, optional
        Whether to convert gas species to ppbV, by default True.
    mech : str, optional
        Mechanism for diagnostics, by default "racm_esrl_vcp".
    var_list : List[str], optional
        List of variables to keep, by default None.
    surf_only : bool, optional
        Keep only surface layer, by default False.
    surf_only_nc : bool, optional
        Whether input is already surface-only, by default False.

    Returns
    -------
    xr.Dataset
        The preprocessed dataset.
    """
    # 1. Coordinate and Dimension Renaming
    rename_dict = {
        "Time": "time",
        "south_north": "y",
        "west_east": "x",
        "XLONG": "longitude",
        "XLAT": "latitude",
        "bottom_top": "z",
        "bottom_top_stag": "z_stag",
        "soil_layers_stag": "z_soil",
    }
    # Only rename what exists
    # We must be careful not to rename if target name already exists as a dimension
    actual_rename = {}
    for k, v in rename_dict.items():
        if k in ds.variables or k in ds.dims:
            if v in ds.dims and k != v:
                # Target exists, maybe it's already renamed or there's a conflict
                continue
            actual_rename[k] = v

    if actual_rename:
        ds = ds.rename(actual_rename)

    # 2. Lazy Time Parsing
    if "Times" in ds.variables:
        ds = _parse_wrf_times(ds)

    # 3. Subset variables if requested
    if var_list is not None:
        # We must keep coordinates and some essentials
        essentials = ["latitude", "longitude", "time", "z", "z_stag", "z_soil"]
        to_keep = set(var_list) | set(essentials)
        available = [v for v in ds.variables if v in to_keep]
        ds = ds[available]

    # 4. Handle Surface Only
    if surf_only and not surf_only_nc and "z" in ds.dims:
        ds = ds.isel(z=[0])

    # 5. Unit Conversions (Lazy)
    if convert_to_ppb:
        ds = _convert_to_ppb(ds)

    # convert "ug/kg-dryair -> ug/m3" if density can be calculated
    ds = _convert_ugkg_to_ugm3(ds)

    # 6. Scientific Hygiene
    ds = ds.reset_coords()
    coords = [c for c in ["latitude", "longitude", "time"] if c in ds.variables]
    ds = ds.set_coords(coords)

    # Strip whitespace from string attributes
    for var in ds.variables:
        for attr, val in ds[var].attrs.items():
            if isinstance(val, str):
                ds[var].attrs[attr] = val.strip()

    # Update history
    history = (
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Preprocessed WRF-Chem data."
    )
    if "history" in ds.attrs:
        ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
    else:
        ds.attrs["history"] = history

    return ds


def _parse_wrf_times(ds: xr.Dataset) -> xr.Dataset:
    """
    Parse WRF 'Times' character array into a 'time' coordinate lazily.
    """
    times_var = ds.Times

    # WRF Times is usually (time, DateStrLen)
    # But it might be (Time, DateStrLen) if not renamed yet,
    # but we just renamed Time -> time.

    # Find the string dimension
    string_dim = [d for d in times_var.dims if d != "time"]
    if not string_dim:
        if times_var.ndim == 1:
            string_dim = [times_var.dims[0]]
        else:
            return ds

    string_dim = string_dim[-1]

    def _parse_times_wrapped(times_arr):
        # times_arr is the core dimension part (DateStrLen)
        # We use a vectorized version to be safe
        def _single_parse(t):
            try:
                # Handle bytes or strings
                if hasattr(t, "tobytes"):
                    s = t.tobytes().decode().strip().replace("_", " ")
                else:
                    s = (
                        "".join([c.decode() if hasattr(c, "decode") else c for c in t])
                        .strip()
                        .replace("_", " ")
                    )
                return np.datetime64(pd.to_datetime(s))
            except Exception:
                return np.datetime64("NaT")

        # If it's more than 1D (because of vectorize=True), np.apply_along_axis is called by apply_ufunc
        return _single_parse(times_arr)

    # To avoid the xarray broadcasting issue, let's try to ensure times_var is a DataArray with correct dims
    parsed_times = xr.apply_ufunc(
        _parse_times_wrapped,
        times_var,
        input_core_dims=[[string_dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.dtype("datetime64[ns]")],
    )

    # Force the name and dims of parsed_times to be 'time' if it was batch-dimensioned by it
    if "time" in parsed_times.dims:
        parsed_times = parsed_times.rename("time")

    # If 'time' dimension already exists, update it
    if "time" in ds.dims:
        # Avoid assign_coords if it causes issues, just set it
        ds.coords["time"] = parsed_times
    else:
        ds["time"] = parsed_times

    return ds


def _convert_to_ppb(ds: xr.Dataset) -> xr.Dataset:
    """
    Lazy conversion of ppmv to ppbV.
    """
    for i in ds.data_vars:
        if "units" in ds[i].attrs:
            units = ds[i].attrs["units"].lower()
            if "ppmv" in units:
                ds[i] = ds[i] * 1000.0
                ds[i].attrs["units"] = "ppbV"
    return ds


def _convert_ugkg_to_ugm3(ds: xr.Dataset) -> xr.Dataset:
    """
    Lazy conversion of ug/kg-dryair to ug/m3.
    """
    if "ALT" in ds.variables:
        for i in ds.data_vars:
            if "units" in ds[i].attrs:
                units = ds[i].attrs["units"].lower()
                if "ug/kg" in units:
                    ds[i] = ds[i] / ds["ALT"]
                    ds[i].attrs["units"] = r"$\mu g m^{-3}$"
    elif "P" in ds.variables and "PB" in ds.variables and "T" in ds.variables:
        R = 287.05
        P_tot = ds["P"] + ds["PB"]
        T_actual = (ds["T"] + 300.0) * (P_tot / 100000.0) ** (287.05 / 1004.5)
        rho = P_tot / (R * T_actual)

        for i in ds.data_vars:
            if "units" in ds[i].attrs:
                units = ds[i].attrs["units"].lower()
                if "ug/kg" in units:
                    ds[i] = ds[i] * rho
                    ds[i].attrs["units"] = r"$\mu g m^{-3}$"

    return ds
