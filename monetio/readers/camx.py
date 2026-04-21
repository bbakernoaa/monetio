"""CAMx Reader"""

from functools import partial
from typing import Any

import numpy as np
import xarray as xr

from monetio.grids import grid_from_dataset

from .base import (
    GriddedReader,
    _add_ioapi_latlon,
    _convert_to_ppb,
    _format_units,
    add_lazy_diagnostic,
    register_reader,
)
from .camx_specs import COARSE, DIAGNOSTICS, FINE, NOY_GAS, POC
from .sat_utils import update_history
from .time_utils import parse_ioapi_times


@register_reader("camx")
class CAMxReader(GriddedReader):
    """
    Reader for CAMx model output files.
    """

    def open_dataset(
        self,
        files: str | list[str],
        earth_radius: float = 6370000,
        convert_to_ppb: bool = True,
        drop_duplicates: bool = False,
        **kwargs: Any,
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
        **kwargs : Any
            Additional arguments passed to the driver.

        Returns
        -------
        xarray.Dataset
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
            ds = update_history(ds, "Dropped duplicate time steps.")

        ds = self.harmonize(ds)

        # Update history
        ds = update_history(ds, "Read CAMx data.")

        return ds


def camx_preprocess(
    ds: xr.Dataset,
    *,
    earth_radius: float = 6370000,
    convert_to_ppb: bool = True,
) -> xr.Dataset:
    """
    Preprocess function for a single CAMx file.

    Parameters
    ----------
    ds : xarray.Dataset
        Input CAMx dataset.
    earth_radius : float, optional
        Earth radius in meters, by default 6370000.
    convert_to_ppb : bool, optional
        Convert gas species to ppbV, by default True.

    Returns
    -------
    xarray.Dataset
        Processed dataset.

    Examples
    --------
    >>> ds = camx_preprocess(ds, convert_to_ppb=True)
    """
    # 1. Add lazy diagnostic variables
    for name, spec in DIAGNOSTICS.items():
        ds = add_lazy_diagnostic(ds, name, spec)

    # 2. Grid and Coordinates
    grid = grid_from_dataset(ds, earth_radius=earth_radius)
    if grid:
        ds = ds.assign_attrs({"proj4_srs": grid})
        ds = _add_ioapi_latlon(ds, grid)

        # Also assign proj4_srs to all data variables for compatibility
        for var in ds.data_vars:
            ds[var].attrs["proj4_srs"] = grid

    # 3. Time
    if "TFLAG" in ds.variables:
        ds = _get_times(ds)

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
    ds = update_history(ds, "Preprocessed CAMx data.")

    return ds


def _get_times(ds: xr.Dataset) -> xr.Dataset:
    """
    Extracts and assigns time coordinate from TFLAG lazily.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.

    Returns
    -------
    xarray.Dataset
        Dataset with 'time' coordinate.

    Examples
    --------
    >>> ds = _get_times(ds)
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

    # Use apply_ufunc to construct dates lazily using vectorized parser
    dates = xr.apply_ufunc(
        parse_ioapi_times,
        tflag.isel(**{dt_dim: 0}),
        tflag.isel(**{dt_dim: 1}),
        vectorize=False,
        dask="parallelized",
        output_dtypes=[np.dtype("datetime64[ns]")],
    )

    ds = ds.assign_coords(TSTEP=dates)
    ds = ds.rename({"TSTEP": "time"})

    # Update history
    ds = update_history(ds, "Optimized time parsing.")

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
