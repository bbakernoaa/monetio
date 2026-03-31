"""CMAQ File Reader"""

from functools import partial
from typing import Any, List, Union

import numpy as np
import xarray as xr

from monetio.grids import grid_from_dataset

from .base import (
    GriddedReader,
    _convert_to_ppb,
    _format_units,
    add_lazy_diagnostic,
    register_reader,
)
from .cmaq_specs import DIAGNOSTICS
from .sat_utils import update_history
from .time_utils import parse_ioapi_times


@register_reader("cmaq")
class CMAQReader(GriddedReader):
    """
    Reader for CMAQ (Community Multiscale Air Quality) model output files.
    """

    def open_dataset(
        self,
        files: Union[str, List[str]],
        earth_radius: float = 6370000,
        convert_to_ppb: bool = True,
        drop_duplicates: bool = False,
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Reads CMAQ netCDF files.

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
            Additional arguments passed to xarray.open_mfdataset or the driver.

        Returns
        -------
        xarray.Dataset
            The processed CMAQ dataset.
        """
        # 1. Setup preprocessing
        if "preprocess" not in kwargs:
            kwargs["preprocess"] = partial(
                cmaq_preprocess,
                earth_radius=earth_radius,
                convert_to_ppb=convert_to_ppb,
            )

        # 2. Open the dataset using standard xarray (via XarrayDriver)
        if "combine" not in kwargs:
            kwargs["combine"] = "nested"
        if "concat_dim" not in kwargs:
            # If we rename in preprocess, we should use 'time'
            # But to be safe with existing logic, we let it be 'TSTEP' if not renamed yet,
            # or 'time' if it is.
            # Actually, preprocess runs BEFORE concatenation.
            kwargs["concat_dim"] = "time"

        ds = self.driver.open(files, **kwargs)

        # 3. Finalize
        if drop_duplicates:
            ds = ds.drop_duplicates("time")
            ds = update_history(ds, "Dropped duplicate time steps.")

        ds = self.harmonize(ds)

        # Update history
        ds = update_history(ds, "Read CMAQ data.")

        return ds

    def harmonize(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Placeholder for future harmonization logic.

        Parameters
        ----------
        ds : xr.Dataset
            CMAQ dataset.

        Returns
        -------
        xr.Dataset
            Harmonized dataset.
        """
        return ds


def cmaq_preprocess(
    ds: xr.Dataset,
    *,
    earth_radius: float = 6370000,
    convert_to_ppb: bool = True,
) -> xr.Dataset:
    """
    Preprocess function for a single CMAQ file.

    Parameters
    ----------
    ds : xarray.Dataset
        Input CMAQ dataset.
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
    >>> ds = cmaq_preprocess(ds, convert_to_ppb=True)
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

    # 6. Scientific Hygiene: Strip whitespace from all string attributes
    for var in ds.variables:
        for attr, val in ds[var].attrs.items():
            if isinstance(val, str):
                ds[var].attrs[attr] = val.strip()

    # Update history
    ds = update_history(ds, "Preprocessed CMAQ data.")

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
    # TFLAG format: [YYYYDDD, HHMMSS]
    # We take the first variable's flags as they are typically identical for all.
    tflag = ds.TFLAG
    if tflag.ndim == 3:
        # (TSTEP, VAR, DATE_TIME) -> (TSTEP, DATE_TIME)
        tflag = tflag.isel(VAR=0, drop=True)

    # Handle different possible names for DATE_TIME dimension (e.g. DATE_TIME or DATE-TIME)
    dt_dims = [d for d in tflag.dims if "DATE" in str(d).upper() and "TIME" in str(d).upper()]
    if not dt_dims:
        # Fallback to last dimension if none matched
        dt_dim = tflag.dims[-1]
    else:
        dt_dim = dt_dims[0]

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


def _get_latlon(ds: xr.Dataset, proj4_srs: str) -> xr.Dataset:
    """
    Assigns latitude and longitude coordinates lazily.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with IOAPI grid metadata.
    proj4_srs : str
        The PROJ4 projection string.

    Returns
    -------
    xarray.Dataset
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

    # 2. Broadcast to 2D (ensure ROW is first dim)
    xda = xr.DataArray(x, dims="COL")
    yda = xr.DataArray(y, dims="ROW")

    if hasattr(ds, "chunks") and ds.chunks:
        # Use chunks from dataset if available
        xda = xda.chunk({"COL": ds.chunks.get("COL", "auto")})
        yda = yda.chunk({"ROW": ds.chunks.get("ROW", "auto")})

    yv, xv = xr.broadcast(yda, xda)

    # 3. Apply projection lazily
    def _proj_inv(x_val: np.ndarray, y_val: np.ndarray, p_srs: str) -> tuple:
        """
        Vectorized inverse projection wrapper.

        Parameters
        ----------
        x_val : np.ndarray
            X coordinates in meters.
        y_val : np.ndarray
            Y coordinates in meters.
        p_srs : str
            PROJ4 projection string.

        Returns
        -------
        tuple
            (longitude, latitude) arrays.
        """
        # Ensure p_srs is a string if it came as an array
        if isinstance(p_srs, (np.ndarray, np.generic)):
            p_srs = p_srs.item()
        if hasattr(p_srs, "decode"):
            p_srs = p_srs.decode()

        p = Proj(p_srs)
        return p(x_val, y_val, inverse=True)

    lon, lat = xr.apply_ufunc(
        _proj_inv,
        xv,
        yv,
        proj4_srs,
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

    # Update history
    ds = update_history(ds, "Generated Latitude/Longitude coordinates.")

    return ds
