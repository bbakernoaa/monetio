"""CMAQ File Reader"""

import datetime
from functools import partial
from typing import List, Union

import xarray as xr
from pandas import Series, to_datetime

from monetio.grids import get_latlon_ioapi, grid_from_dataset

from .base import GriddedReader, register_reader
from .cmaq_specs import DIAGNOSTICS, DiagnosticSpec


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
        history = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read CMAQ data."
        if "history" in ds.attrs:
            ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
        else:
            ds.attrs["history"] = history

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
    Extracts and assigns time coordinate from TFLAG.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    drop_duplicates : bool, optional
        Whether to drop duplicate time steps, by default False.

    Returns
    -------
    xr.Dataset
        Dataset with 'time' coordinate instead of 'TSTEP'.
    """
    # TFLAG format: [YYYYDDD, HHMMSS]
    # We take the first variable's flags as they are typically identical for all.
    tflag = ds.TFLAG.compute()
    if tflag.ndim == 3:
        # (TSTEP, VAR, DATE_TIME) -> (TSTEP, DATE_TIME)
        tflag = tflag[:, 0, :]

    tflag1 = Series(tflag[:, 0]).astype(str).str.zfill(7)
    tflag2 = Series(tflag[:, 1]).astype(str).str.zfill(6)

    dates = to_datetime(tflag1 + tflag2, format="%Y%j%H%M%S")

    if drop_duplicates:
        # Use pandas to find unique indices, keeping the last occurrence (common for CMAQ)
        unique_indices = Series(dates).drop_duplicates(keep="last").index.values
        ds = ds.isel(TSTEP=unique_indices)
        dates = dates[unique_indices]

    ds = ds.assign_coords(TSTEP=dates)
    ds = ds.rename({"TSTEP": "time"})
    return ds


def _get_latlon(ds: xr.Dataset, proj4_srs: str) -> xr.Dataset:
    """
    Assigns latitude and longitude coordinates using the projection string.

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
    # get_latlon_ioapi currently returns NumPy arrays.
    lon, lat = get_latlon_ioapi(ds, proj4_srs)

    ds = ds.assign_coords(longitude=(("ROW", "COL"), lon), latitude=(("ROW", "COL"), lat))
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
