"""CAMx Reader"""

from functools import partial
from typing import Any, List, Optional, Union

import numpy as np
import xarray as xr

from monetio.grids import grid_from_dataset

from .base import GriddedReader, register_reader
from .camx_specs import COARSE, DIAGNOSTICS, FINE, NOY_GAS, POC, DiagnosticSpec
from .sat_utils import update_history
from .time_utils import parse_ioapi_times


@register_reader("camx")
class CAMxReader(GriddedReader):
    """
    Reader for CAMx model output files.
    """

    def open_dataset(
        self,
        files: Optional[Union[str, List[str]]] = None,
        dates: Optional[Any] = None,
        earth_radius: float = 6370000,
        convert_to_ppb: bool = True,
        drop_duplicates: bool = False,
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Reads CAMx netCDF files.

        Parameters
        ----------
        files : Union[str, List[str]], optional
            File path, list of paths, or glob pattern.
        dates : Any, optional
            Dates to retrieve if files are not provided.
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

        ds = super().open_dataset(
            files,
            dates,
            earth_radius=earth_radius,
            convert_to_ppb=convert_to_ppb,
            drop_duplicates=drop_duplicates,
            **kwargs,
        )

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

    # 2. Check for pre-calculated summary variables to prevent regressions
    aliases = {
        "PM25": ["PM25_TOT", "PM2_5"],
        "PM10": ["PM_TOT", "PM10"],
    }

    for alias in aliases.get(name, []):
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


def _get_latlon(ds: xr.Dataset, proj4_srs: str) -> xr.Dataset:
    """
    Assigns latitude and longitude coordinates lazily.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.
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

    # 2. Broadcast to 2D
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
    ds : xarray.Dataset
        Input dataset.

    Returns
    -------
    xarray.Dataset
        Dataset with converted units.
    """
    to_convert = [
        v for v in ds.data_vars if "units" in ds[v].attrs and "ppm" in ds[v].attrs["units"].lower()
    ]

    if not to_convert:
        return ds

    for v in to_convert:
        ds[v] = ds[v] * 1000.0
        ds[v].attrs["units"] = "ppbV"

    # Update history
    ds = update_history(ds, f"Converted {', '.join(to_convert)} from ppmV to ppbV.")

    return ds


def _format_units(ds: xr.Dataset) -> xr.Dataset:
    """
    Formats unit strings for particulate matter.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.

    Returns
    -------
    xarray.Dataset
        Dataset with formatted unit strings.
    """
    to_format = [
        v
        for v in ds.data_vars
        if "units" in ds[v].attrs
        and ("micrograms" in ds[v].attrs["units"].lower() or "ug" in ds[v].attrs["units"].lower())
    ]

    if not to_format:
        return ds

    for v in to_format:
        ds[v].attrs["units"] = r"$\mu g m^{-3}$"

    # Update history
    ds = update_history(ds, rf"Formatted units for {', '.join(to_format)} to $\mu g m^{{-3}}$.")

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
