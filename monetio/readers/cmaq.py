"""CMAQ File Reader"""

import datetime
from typing import List, Union

import xarray as xr
from pandas import DatetimeIndex, Series, to_datetime

from monetio.grids import get_latlon_ioapi, grid_from_dataset

from .base import GriddedReader, register_reader
from .cmaq_specs import DIAGNOSTICS


@register_reader("cmaq")
class CMAQReader(GriddedReader):
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
            Drop duplicate time steps, by default False.
        **kwargs : dict
            Additional arguments passed to xarray.open_mfdataset or the driver.

        Returns
        -------
        xr.Dataset
            The processed CMAQ dataset.
        """
        # 1. Open the dataset using standard xarray (Lazy loading)

        # We ensure standard CMAQ combination logic is present
        if "combine" not in kwargs:
            kwargs["combine"] = "nested"
        if "concat_dim" not in kwargs:
            kwargs["concat_dim"] = "TSTEP"

        # Use cmaq_preprocess to add lazy diagnostic variables
        kwargs["preprocess"] = cmaq_preprocess

        ds = self.driver.open(files, **kwargs)

        # 2. Pre-processing specific to CMAQ (Global)

        # get the grid information
        grid = grid_from_dataset(ds, earth_radius=earth_radius)

        # assign attributes for dataset and all DataArrays
        ds = ds.assign_attrs({"proj4_srs": grid})
        for i in ds.variables:
            ds[i] = ds[i].assign_attrs({"proj4_srs": grid})
            for j in ds[i].attrs:
                # Strip whitespace from attributes
                if isinstance(ds[i].attrs[j], str):
                    ds[i].attrs[j] = ds[i].attrs[j].strip()

        # get the times
        if "TFLAG" in ds.variables or "TFLAG" in ds.coords:
            ds = _get_times(ds, drop_duplicates=drop_duplicates)

        # get the lat lon
        ds = _get_latlon(ds, grid)

        # rename dimensions
        ds = ds.rename({"COL": "x", "ROW": "y", "LAY": "z"})

        # convert all gas species to ppbv
        if convert_to_ppb:
            ds = _convert_to_ppb(ds)

        # convert 'micrograms to \mu g'
        ds = _format_units(ds)

        # 3. Harmonize (Standardize names)
        ds = self.harmonize(ds)

        # Update history
        history = (
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read CMAQ data."
        )
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


def cmaq_preprocess(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess function to add lazy diagnostic variables.
    Can be passed to xarray.open_mfdataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input CMAQ dataset.

    Returns
    -------
    xr.Dataset
        Dataset with diagnostic variables added.
    """
    for name, spec in DIAGNOSTICS.items():
        ds = add_lazy_diagnostic(ds, name, spec)
    return ds


def add_lazy_diagnostic(ds: xr.Dataset, name: str, spec: any) -> xr.Dataset:
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


def _get_times(d: xr.Dataset, drop_duplicates: bool) -> xr.Dataset:
    """
    Extracts and assigns time coordinate from TFLAG.

    Parameters
    ----------
    d : xr.Dataset
        Input dataset.
    drop_duplicates : bool
        Whether to drop duplicate time steps.

    Returns
    -------
    xr.Dataset
        Dataset with time coordinate.
    """
    # TFLAG processing is inherently eager as it involves coordinate construction
    # but we follow the protocol by avoiding explicit .values where possible.
    tflag = d.TFLAG.compute()  # Explicit compute here as we are building coordinates
    idims = len(tflag.dims)
    if idims == 2:
        tflag1 = Series(tflag[:, 0]).astype(str).str.zfill(7)
        tflag2 = Series(tflag[:, 1]).astype(str).str.zfill(6)
    else:
        tflag1 = Series(tflag[:, 0, 0]).astype(str).str.zfill(7)
        tflag2 = Series(tflag[:, 0, 1]).astype(str).str.zfill(6)

    date = to_datetime([i + j for i, j in zip(tflag1, tflag2)], format="%Y%j%H%M%S")
    if drop_duplicates:
        series_date = Series(date)
        indexdates = series_date.drop_duplicates(keep="last").index.values
        d = d.isel(TSTEP=indexdates)
        d["TSTEP"] = DatetimeIndex(date[indexdates])
    else:
        d["TSTEP"] = DatetimeIndex(date)
    return d.rename({"TSTEP": "time"})


def _get_latlon(dset: xr.Dataset, proj4_srs: str) -> xr.Dataset:
    """
    Assigns latitude and longitude using pyproj based on the projection string.

    Parameters
    ----------
    dset : xr.Dataset
        Input dataset with IOAPI grid metadata (XORIG, YORIG, XCELL, YCELL, NCOLS, NROWS).
    proj4_srs : str
        The PROJ4 projection string.

    Returns
    -------
    xr.Dataset
        Dataset with latitude and longitude coordinates.
    """
    lon, lat = get_latlon_ioapi(dset, proj4_srs)

    # We do not flip here because our meshgrid is already bottom-to-top (increasing y),
    # matching the IOAPI/CMAQ convention and the original behavior when flipping pyresample's top-to-bottom output.
    dset["longitude"] = xr.DataArray(lon, dims=["ROW", "COL"])
    dset["latitude"] = xr.DataArray(lat, dims=["ROW", "COL"])
    dset = dset.assign_coords(longitude=dset.longitude, latitude=dset.latitude)
    return dset


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
