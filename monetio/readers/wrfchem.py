"""WRF-Chem Reader"""

from functools import partial
from typing import List, Optional, Union

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
    ds : xr.Dataset
        Input dataset.
    name : str
        Name of the diagnostic variable.
    spec : DiagnosticSpec
        Specification for the diagnostic.

    Returns
    -------
    xr.Dataset
        Dataset with diagnostic added if possible.
    """
    # Check if name already exists as a data variable (pre-calculated in WRF)
    # Some modules output PM2_5_DRY or similar.
    wrf_aliases = {
        "PM25": ["PM2_5_DRY", "PM25_TOT"],
        "PM10": ["PM10_DRY", "PM10_TOT"],
    }

    aliases = wrf_aliases.get(name, [])
    for alias in aliases:
        if alias in ds.data_vars:
            ds[name] = ds[alias].copy()
            ds[name].attrs.update(
                {"units": spec.units, "name": spec.name, "long_name": spec.long_name}
            )
            return ds

    available_vars = [v for v in spec.variables if v in ds.data_vars]
    if not available_vars:
        return ds

    # If weights are provided, they must match the full variable list in spec
    if spec.weights is not None:
        weights_map = dict(zip(spec.variables, spec.weights))
        weights = [weights_map[v] for v in available_vars]
    else:
        weights = [1.0] * len(available_vars)

    # Compute lazy sum
    with xr.set_options(keep_attrs=True):
        new_var = ds[available_vars[0]] * weights[0]
        for i in range(1, len(available_vars)):
            new_var = new_var + ds[available_vars[i]] * weights[i]

    # Inherit units from constituent variables if available, otherwise use spec
    units = ds[available_vars[0]].attrs.get("units", spec.units)

    ds[name] = new_var.assign_attrs(
        {"units": units, "name": spec.name, "long_name": spec.long_name}
    )

    # Update history
    ds = update_history(ds, f"Added lazy diagnostic: {name}")

    return ds


def _parse_wrf_times(ds: xr.Dataset) -> xr.Dataset:
    """
    Parse WRF 'Times' character array into a 'time' coordinate lazily.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with 'Times' variable.

    Returns
    -------
    xr.Dataset
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
    ds : xr.Dataset
        Input dataset.

    Returns
    -------
    xr.Dataset
        Dataset with converted units.
    """
    converted = False
    for i in ds.data_vars:
        if "units" in ds[i].attrs:
            units = ds[i].attrs["units"].lower()
            if "ppmv" in units:
                ds[i] = ds[i] * 1000.0
                ds[i].attrs["units"] = "ppbV"
                converted = True

    if converted:
        ds = update_history(ds, "Converted ppmV to ppbV.")

    return ds


def _convert_ugkg_to_ugm3(ds: xr.Dataset) -> xr.Dataset:
    """
    Lazy conversion of ug/kg-dryair to ug/m3.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.

    Returns
    -------
    xr.Dataset
        Dataset with converted units if ALT or P/PB/T are available.
    """
    converted = False
    if "ALT" in ds.variables:
        for i in ds.data_vars:
            if "units" in ds[i].attrs:
                units = ds[i].attrs["units"].lower()
                if "ug/kg" in units:
                    ds[i] = ds[i] / ds["ALT"]
                    ds[i].attrs["units"] = r"$\mu g m^{-3}$"
                    converted = True
    elif "P" in ds.variables and "PB" in ds.variables and "T" in ds.variables:
        # Standard WRF-Chem density calculation
        # P_tot is total pressure (perturbation + base)
        # T_actual is temperature in K
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
                    converted = True

    if converted:
        ds = update_history(ds, r"Converted $\mu g/kg$ to $\mu g/m^3$ using air density.")

    return ds
