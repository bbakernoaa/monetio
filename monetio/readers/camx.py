"""CAMx Reader"""

import datetime
from functools import partial
from typing import List, Union

import numpy as np
import pandas as pd
import xarray as xr
from pandas import Series

from monetio.grids import grid_from_dataset

from .base import GriddedReader, register_reader
from .camx_specs import COARSE, DIAGNOSTICS, FINE, NOY_GAS, POC, DiagnosticSpec


@register_reader("camx")
class CAMxReader(GriddedReader):
    """
    Reader for CAMx model output files.
    """

    def open_dataset(
        self,
        files: Union[str, List[str]],
        earth_radius: float = 6370000,
        convert_to_ppb: bool = True,
        drop_duplicates: bool = False,
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads CAMx netCDF files.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path, list of paths, or glob pattern.
        earth_radius : float, optional
            Earth radius in meters, by default 6370000.
        convert_to_ppb : bool, optional
            Convert gas species from ppmV to ppbV, by default True.
        drop_duplicates : bool, optional
            Drop duplicate time steps within each file, by default False.
        **kwargs : dict
            Additional arguments passed to the driver.

        Returns
        -------
        xr.Dataset
            The processed CAMx dataset.
        """
        # Set default backend kwargs for CAMx if not present
        if "engine" not in kwargs:
            kwargs["engine"] = "pseudonetcdf"
            if "backend_kwargs" not in kwargs:
                kwargs["backend_kwargs"] = {"format": "uamiv"}

        # 1. Setup preprocessing
        if "preprocess" not in kwargs:
            kwargs["preprocess"] = partial(
                camx_preprocess,
                earth_radius=earth_radius,
                convert_to_ppb=convert_to_ppb,
                drop_duplicates=drop_duplicates,
            )

        # 2. Open the dataset using standard xarray (via XarrayDriver)
        if "combine" not in kwargs:
            kwargs["combine"] = "nested"
        if "concat_dim" not in kwargs:
            kwargs["concat_dim"] = "time"

        ds = self.driver.open(files, **kwargs)

        # 3. Finalize
        if drop_duplicates:
            ds = ds.drop_duplicates("time")

        ds = self.harmonize(ds)

        # Update history
        history = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read CAMx data."
        if "history" in ds.attrs:
            ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
        else:
            ds.attrs["history"] = history

        return ds


def camx_preprocess(
    ds: xr.Dataset,
    *,
    earth_radius: float = 6370000,
    convert_to_ppb: bool = True,
    drop_duplicates: bool = False,
) -> xr.Dataset:
    """
    Preprocess function for a single CAMx file.

    Parameters
    ----------
    ds : xr.Dataset
        Input CAMx dataset.
    earth_radius : float, optional
        Earth radius in meters, by default 6370000.
    convert_to_ppb : bool, optional
        Convert gas species to ppbV, by default True.
    drop_duplicates : bool, optional
        Drop duplicate time steps, by default False.

    Returns
    -------
    xr.Dataset
        Processed dataset.
    """
    # 1. Add lazy diagnostic variables
    for name, spec in DIAGNOSTICS.items():
        ds = add_lazy_diagnostic(ds, name, spec)

    # 2. Grid and Coordinates
    grid = grid_from_dataset(ds, earth_radius=earth_radius)
    if grid:
        ds = ds.assign_attrs({"proj4_srs": grid})
        ds = _get_latlon(ds, grid)

        # Also assign proj4_srs to all data variables for compatibility
        for var in ds.data_vars:
            ds[var].attrs["proj4_srs"] = grid

    # 3. Time
    if "TFLAG" in ds.variables:
        ds = _get_times(ds, drop_duplicates=drop_duplicates)

    # 4. Units and Formatting
    if convert_to_ppb:
        ds = _convert_to_ppb(ds)
    ds = _format_units(ds)

    # 5. Rename dimensions
    rename_dict = {}
    if "COL" in ds.dims:
        rename_dict["COL"] = "x"
    if "ROW" in ds.dims:
        rename_dict["ROW"] = "y"
    if "LAY" in ds.dims:
        rename_dict["LAY"] = "z"
    if rename_dict:
        ds = ds.rename(rename_dict)

    # 6. Predefined mapping tables (backward compatibility)
    ds = _predefined_mapping_tables(ds)

    # 7. Scientific Hygiene: Strip whitespace from all string attributes
    for var in ds.variables:
        for attr, val in ds[var].attrs.items():
            if isinstance(val, str):
                ds[var].attrs[attr] = val.strip()

    # Update history
    history = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Preprocessed CAMx data."
    if "history" in ds.attrs:
        ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
    else:
        ds.attrs["history"] = history

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
        Specification for the diagnostic (from camx_specs).

    Returns
    -------
    xr.Dataset
        Dataset with diagnostic added if possible.
    """
    # Special cases for CAMx pre-existing totals
    if name == "PM25" and "PM25_TOT" in ds.data_vars:
        ds["PM25"] = ds["PM25_TOT"]
        return ds
    if name == "PM10" and "PM_TOT" in ds.data_vars:
        ds["PM10"] = ds["PM_TOT"]
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
    new_var = ds[available_vars[0]] * weights[0]
    for i in range(1, len(available_vars)):
        new_var = new_var + ds[available_vars[i]] * weights[i]

    ds[name] = new_var.assign_attrs(
        {"units": spec.units, "name": spec.name, "long_name": spec.long_name}
    )
    return ds


def _get_times(ds: xr.Dataset, *, drop_duplicates: bool = False) -> xr.Dataset:
    """
    Extracts and assigns time coordinate from TFLAG lazily.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    drop_duplicates : bool, optional
        Whether to drop duplicate time steps, by default False.

    Returns
    -------
    xr.Dataset
        Dataset with 'time' coordinate.
    """
    tflag = ds.TFLAG
    # CAMx TFLAG can be (TSTEP, DATE_TIME) or (TSTEP, VAR, DATE_TIME)
    if tflag.ndim == 3:
        tflag = tflag.isel(VAR=0, drop=True)

    # Handle dimension names (COL is used for DATE_TIME in pseudonetcdf format)
    # Actually it is usually TSTEP and something else.
    # In _get_times from legacy: d["TFLAG"][:, 0]
    # So the last dimension is the DATE_TIME one.
    dt_dim = tflag.dims[-1]

    def _parse_camx_times(yyyymmdd, hhmmss):
        s1 = str(yyyymmdd).zfill(7)
        s2 = str(hhmmss).zfill(6)
        return pd.to_datetime(s1 + s2, format="%Y%j%H%M%S").to_datetime64()

    dates = xr.apply_ufunc(
        _parse_camx_times,
        tflag.isel(**{dt_dim: 0}),
        tflag.isel(**{dt_dim: 1}),
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.dtype("datetime64[ns]")],
    )

    if drop_duplicates:
        dates_computed = dates.compute()
        unique_indices = Series(dates_computed).drop_duplicates(keep="last").index.values
        ds = ds.isel(TSTEP=unique_indices)
        dates = dates.isel(TSTEP=unique_indices)

    ds = ds.assign_coords(TSTEP=dates)
    ds = ds.rename({"TSTEP": "time"})
    return ds


def _get_latlon(ds: xr.Dataset, proj4_srs: str) -> xr.Dataset:
    """
    Assigns latitude and longitude coordinates lazily.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    proj4_srs : str
        The PROJ4 projection string.

    Returns
    -------
    xr.Dataset
        Dataset with 'latitude' and 'longitude' coordinates.
    """
    from pyproj import Proj

    # 1. Generate 1D x and y values
    x = np.linspace(
        ds.XORIG + ds.XCELL * 0.5,
        ds.XORIG + (ds.NCOLS - 0.5) * ds.XCELL,
        ds.NCOLS,
    )
    y = np.linspace(
        ds.YORIG + ds.YCELL * 0.5,
        ds.YORIG + (ds.NROWS - 0.5) * ds.YCELL,
        ds.NROWS,
    )

    # 2. Broadcast to 2D
    yv, xv = xr.broadcast(xr.DataArray(y, dims="ROW"), xr.DataArray(x, dims="COL"))

    # 3. Apply projection lazily
    def _proj_inv(x_val, y_val, p_srs):
        p = Proj(p_srs)
        return p(x_val, y_val, inverse=True)

    lon, lat = xr.apply_ufunc(
        _proj_inv,
        xv,
        yv,
        proj4_srs,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float],
        output_core_dims=[(), ()],
    )

    ds = ds.assign_coords(
        longitude=lon.assign_attrs(
            {"long_name": "Longitude", "units": "degree_east", "standard_name": "longitude"}
        ),
        latitude=lat.assign_attrs(
            {"long_name": "Latitude", "units": "degree_north", "standard_name": "latitude"}
        ),
    )
    return ds


def _convert_to_ppb(ds: xr.Dataset) -> xr.Dataset:
    """
    Converts gas species units from ppmV to ppbV.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.

    Returns
    -------
    xr.Dataset
        Dataset with converted units.
    """
    for i in ds.data_vars:
        if "units" in ds[i].attrs:
            if "ppm" in ds[i].attrs["units"].lower():
                ds[i] = ds[i] * 1000.0
                ds[i].attrs["units"] = "ppbV"
    return ds


def _format_units(ds: xr.Dataset) -> xr.Dataset:
    """
    Formats unit strings for particulate matter.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.

    Returns
    -------
    xr.Dataset
        Dataset with formatted unit strings.
    """
    for i in ds.data_vars:
        if "units" in ds[i].attrs:
            if "micrograms" in ds[i].attrs["units"].lower() or "ug" in ds[i].attrs["units"].lower():
                ds[i].attrs["units"] = r"$\mu g m^{-3}$"
    return ds


def _predefined_mapping_tables(ds: xr.Dataset) -> xr.Dataset:
    """
    Adds mapping tables for backward compatibility.
    """
    # Ported from legacy code
    to_aqs = {
        "OZONE": ["O3"],
        "PM2.5": ["PM25"],
        "CO": ["CO"],
        "NOY": [
            "NO",
            "NO2",
            "NO3",
            "N2O5",
            "HONO",
            "HNO3",
            "PAN",
            "PANX",
            "PNA",
            "NTR",
            "CRON",
            "CRN2",
            "CRNO",
            "CRPX",
            "OPAN",
        ],
        "NOX": ["NO", "NOX"],
        "SO2": ["SO2"],
        "NO": ["NO"],
        "NO2": ["NO2"],
        "SO4f": ["PSO4"],
        "PM10": ["PM10"],
        "NO3f": ["PNO3"],
        "ECf": ["PEC"],
        "OCf": ["OC"],
        "ETHANE": ["ETHA"],
        "BENZENE": ["BENZENE"],
        "TOLUENE": ["TOL"],
        "ISOPRENE": ["ISOP"],
        "O-XYLENE": ["XYL"],
        "WS": ["WSPD10"],
        "TEMP": ["TEMP2"],
        "WD": ["WDIR10"],
        "NAf": ["NA"],
        "NH4f": ["PNH4"],
    }
    # Duplicate for AirNow
    to_airnow = to_aqs.copy()

    mapping_tables = {
        "improve": {},
        "aqs": to_aqs,
        "airnow": to_airnow,
        "crn": {},
        "cems": {},
        "nadp": {},
        "aeronet": {},
    }
    ds = ds.assign_attrs({"mapping_tables": mapping_tables})
    return ds


# Legacy aliases for backward compatibility
fine = FINE
coarse = COARSE
noy_gas = NOY_GAS
poc = POC


def add_lazy_pm25(ds):
    return add_lazy_diagnostic(ds, "PM25", DIAGNOSTICS["PM25"])


def add_lazy_pm10(ds):
    return add_lazy_diagnostic(ds, "PM10", DIAGNOSTICS["PM10"])


def add_lazy_pm_coarse(ds):
    return add_lazy_diagnostic(ds, "PM_COARSE", DIAGNOSTICS["PM_COARSE"])


def add_lazy_noy(ds):
    return add_lazy_diagnostic(ds, "NOy", DIAGNOSTICS["NOy"])


def add_lazy_nox(ds):
    return add_lazy_diagnostic(ds, "NOx", DIAGNOSTICS["NOx"])


def add_multiple_lazy(dset, variables, weights=None):
    from numpy import ones

    if weights is None:
        weights = ones(len(variables))
    new = dset[variables[0]] * weights[0]
    for i in range(1, len(variables)):
        new = new + dset[variables[i]] * weights[i]
    return new
