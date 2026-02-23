"""CMAQ File Reader"""

from functools import partial
from typing import List, Union

import numpy as np
import xarray as xr
from pandas import Series

from monetio.grids import grid_from_dataset

from .base import GriddedReader, register_reader
from .cmaq_specs import DIAGNOSTICS, DiagnosticSpec
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
        **kwargs,
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
        **kwargs : dict
            Additional arguments passed to xarray.open_mfdataset or the driver.

        Returns
        -------
        xr.Dataset
            The processed CMAQ dataset.
        """
        # 1. Setup preprocessing
        if "preprocess" not in kwargs:
            kwargs["preprocess"] = partial(
                cmaq_preprocess,
                earth_radius=earth_radius,
                convert_to_ppb=convert_to_ppb,
                drop_duplicates=drop_duplicates,
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
    drop_duplicates: bool = False,
) -> xr.Dataset:
    """
    Preprocess function for a single CMAQ file.

    Parameters
    ----------
    ds : xr.Dataset
        Input CMAQ dataset.
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

    # 6. Scientific Hygiene: Strip whitespace from all string attributes
    for var in ds.variables:
        for attr, val in ds[var].attrs.items():
            if isinstance(val, str):
                ds[var].attrs[attr] = val.strip()

    # Update history
    ds = update_history(ds, "Preprocessed CMAQ data.")

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
        Specification for the diagnostic (from cmaq_specs).

    Returns
    -------
    xr.Dataset
        Dataset with diagnostic added if possible.
    """
    # Check for pre-existing summary variables to prevent regressions
    if name == "PM25" and "PM25_TOT" in ds.data_vars:
        ds["PM25"] = ds["PM25_TOT"]
        return ds
    if name == "PM10" and "PMC_TOT" in ds.data_vars:
        ds["PM10"] = ds["PMC_TOT"]
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

    if drop_duplicates:
        # Warning: drop_duplicates requires computation of the coordinate
        # to identify unique values. This is an unavoidable "Lazy Breaker"
        # but we only trigger it if explicitly requested.
        dates_computed = dates.compute()
        unique_indices = Series(dates_computed).drop_duplicates(keep="last").index.values
        ds = ds.isel(TSTEP=unique_indices)
        dates = dates.isel(TSTEP=unique_indices)

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
    ds : xr.Dataset
        Input dataset with IOAPI grid metadata.
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
            if "ppmV" in ds[i].attrs["units"]:
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
            if "micrograms" in ds[i].attrs["units"]:
                ds[i].attrs["units"] = r"$\mu g m^{-3}$"
    return ds
