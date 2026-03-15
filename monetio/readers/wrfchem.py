"""WRF-Chem Reader"""

from functools import partial
from typing import Any, List, Optional, Union

import numpy as np
import xarray as xr

from .base import GriddedReader, register_reader
from .sat_utils import update_history
from .time_utils import parse_wrf_times
from .wrfchem_specs import DIAGNOSTICS, DiagnosticSpec


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
        **kwargs: Any,
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
        var_list : list of str, optional
            List of variables to include, by default None.
        surf_only : bool, optional
            Whether to only keep surface data, by default False.
        surf_only_nc : bool, optional
            Whether input data already contains only surface data, by default False.
        **kwargs : Any
            Additional arguments passed to the driver.

        Returns
        -------
        xarray.Dataset
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
        ds = update_history(ds, "Read WRF-Chem data.")

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
    Preprocess function for a single WRF-Chem file.

    Parameters
    ----------
    ds : xarray.Dataset
        Input WRF-Chem dataset.
    convert_to_ppb : bool, optional
        Whether to convert gas species to ppbV, by default True.
    mech : str, optional
        Mechanism for diagnostics, by default "racm_esrl_vcp".
    var_list : list of str, optional
        List of variables to keep, by default None.
    surf_only : bool, optional
        Keep only surface layer, by default False.
    surf_only_nc : bool, optional
        Whether input is already surface-only, by default False.

    Returns
    -------
    xarray.Dataset
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

    # 3. Unit Conversions (Lazy) - Move before diagnostics to ensure synced units
    if convert_to_ppb:
        ds = _convert_to_ppb(ds)

    # convert "ug/kg-dryair -> ug/m3" if density can be calculated
    ds = _convert_ugkg_to_ugm3(ds)

    # 4. Add lazy diagnostic variables
    for name, spec in DIAGNOSTICS.items():
        ds = add_lazy_diagnostic(ds, name, spec)

    # 5. Subset variables if requested
    if var_list is not None:
        # We must keep coordinates and some essentials
        essentials = ["latitude", "longitude", "time", "z", "z_stag", "z_soil"]
        to_keep = set(var_list) | set(essentials)
        # Add those that were added as diagnostics
        to_keep |= {name for name in DIAGNOSTICS if name in ds.variables}
        available = [v for v in ds.variables if v in to_keep]
        ds = ds[available]

    # 6. Handle Surface Only
    if surf_only and not surf_only_nc and "z" in ds.dims:
        ds = ds.isel(z=[0])

    # 7. Scientific Hygiene
    ds = ds.reset_coords()
    coords = [c for c in ["latitude", "longitude", "time"] if c in ds.variables]
    ds = ds.set_coords(coords)

    # Strip whitespace from string attributes
    for var in ds.variables:
        for attr, val in ds[var].attrs.items():
            if isinstance(val, str):
                ds[var].attrs[attr] = val.strip()

    # Update history
    ds = update_history(ds, "Preprocessed WRF-Chem data.")

    return ds


def add_lazy_diagnostic(ds: xr.Dataset, name: str, spec: DiagnosticSpec) -> xr.Dataset:
    """
    Adds a lazy diagnostic variable to the dataset if constituent variables exist.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.
    name : str
        Name of the diagnostic variable.
    spec : DiagnosticSpec
        Specification for the diagnostic.

    Returns
    -------
    xarray.Dataset
        Dataset with diagnostic added if possible.
    """
    # 1. Check if name already exists as a data variable
    if name in ds.data_vars:
        return ds

    # 2. Check for pre-calculated WRF-Chem aliases (e.g. PM2_5_DRY)
    wrf_aliases = {
        "PM25": ["PM2_5_DRY", "PM25_TOT", "PM2_5"],
        "PM10": ["PM10_DRY", "PM10_TOT", "PM10"],
    }

    aliases = wrf_aliases.get(name, [])
    for alias in aliases:
        if alias in ds.data_vars:
            ds[name] = ds[alias].copy()
            ds[name].attrs.update(
                {"units": spec.units, "name": spec.name, "long_name": spec.long_name}
            )
            # Update history
            ds = update_history(ds, f"Added lazy diagnostic: {name} (using alias {alias}).")
            return ds

    # 3. Identify constituent variables available in the dataset
    available_vars = [v for v in spec.variables if v in ds.data_vars]
    if not available_vars:
        return ds

    # If weights are provided, they must match the full variable list in spec
    if spec.weights is not None:
        weights_map = dict(zip(spec.variables, spec.weights))
        weights = [weights_map[v] for v in available_vars]
    else:
        weights = [1.0] * len(available_vars)

    # 4. Compute lazy sum with unit synchronization
    with xr.set_options(keep_attrs=True):
        # Use first variable as base
        v0 = available_vars[0]
        new_var = ds[v0] * weights[0]
        base_units = ds[v0].attrs.get("units", "").lower()

        for i in range(1, len(available_vars)):
            v = available_vars[i]
            v_var = ds[v]
            v_units = v_var.attrs.get("units", "").lower()

            # Unit synchronization (e.g. ppmV vs ppbV)
            if v_units != base_units:
                if "ppm" in v_units and "ppb" in base_units:
                    v_var = v_var * 1000.0
                elif "ppb" in v_units and "ppm" in base_units:
                    v_var = v_var / 1000.0

            new_var = new_var + v_var * weights[i]

    # Inherit units from constituent variables if available, otherwise use spec
    units = ds[v0].attrs.get("units", spec.units)

    ds[name] = new_var.assign_attrs(
        {"units": units, "name": spec.name, "long_name": spec.long_name}
    )

    # Update history
    ds = update_history(ds, f"Added lazy diagnostic: {name} (sum of {', '.join(available_vars)}).")

    return ds


def _parse_wrf_times(ds: xr.Dataset) -> xr.Dataset:
    """
    Parse WRF 'Times' character array into a 'time' coordinate lazily.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with 'Times' variable.

    Returns
    -------
    xarray.Dataset
        Dataset with 'time' coordinate.
    """
    times_var = ds.Times

    # Find the string dimension
    string_dim = [d for d in times_var.dims if d != "time"]
    if not string_dim:
        if times_var.ndim == 1:
            string_dim = [times_var.dims[0]]
        else:
            return ds

    string_dim = string_dim[-1]

    # Use vectorized parser from time_utils
    parsed_times = xr.apply_ufunc(
        parse_wrf_times,
        times_var,
        input_core_dims=[[string_dim]],
        output_core_dims=[[]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[np.dtype("datetime64[ns]")],
    )

    # Force the name and dims of parsed_times to be 'time' if it was batch-dimensioned by it
    if "time" in parsed_times.dims:
        parsed_times = parsed_times.rename("time")

    # If 'time' dimension already exists, update it
    if "time" in ds.dims:
        ds.coords["time"] = parsed_times
    else:
        ds["time"] = parsed_times

    # Update history
    ds = update_history(ds, "Optimized time parsing.")

    return ds


def _convert_to_ppb(ds: xr.Dataset) -> xr.Dataset:
    """
    Lazy conversion of ppmv to ppbV.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.

    Returns
    -------
    xarray.Dataset
        Dataset with converted units.
    """
    to_convert = [
        v for v in ds.data_vars if "units" in ds[v].attrs and "ppmv" in ds[v].attrs["units"].lower()
    ]

    if not to_convert:
        return ds

    for v in to_convert:
        ds[v] = ds[v] * 1000.0
        ds[v].attrs["units"] = "ppbV"

    # Update history
    ds = update_history(ds, f"Converted {', '.join(to_convert)} from ppmV to ppbV.")

    return ds


def _convert_ugkg_to_ugm3(ds: xr.Dataset) -> xr.Dataset:
    """
    Lazy conversion of ug/kg-dryair to ug/m3.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.

    Returns
    -------
    xarray.Dataset
        Dataset with converted units if ALT or P/PB/T are available.
    """
    to_convert = [
        v
        for v in ds.data_vars
        if "units" in ds[v].attrs and "ug/kg" in ds[v].attrs["units"].lower()
    ]

    if not to_convert:
        return ds

    method = None
    if "ALT" in ds.variables:
        # Use inverse density (specific volume)
        for v in to_convert:
            ds[v] = ds[v] / ds["ALT"]
            ds[v].attrs["units"] = r"$\mu g m^{-3}$"
        method = "using ALT (specific volume)"
    elif all(k in ds.variables for k in ["P", "PB", "T"]):
        # Standard WRF-Chem density calculation
        # P_tot is total pressure (perturbation + base)
        # T_actual is temperature in K
        R = 287.05
        P_tot = ds["P"] + ds["PB"]
        # Potential temperature to actual temperature conversion
        T_actual = (ds["T"] + 300.0) * (P_tot / 100000.0) ** (287.05 / 1004.5)
        rho = P_tot / (R * T_actual)

        for v in to_convert:
            ds[v] = ds[v] * rho
            ds[v].attrs["units"] = r"$\mu g m^{-3}$"
        method = "using air density calculated from P, PB, T"

    if method:
        ds = update_history(
            ds, rf"Converted {', '.join(to_convert)} from $\mu g/kg$ to $\mu g/m^3$ {method}."
        )

    return ds
