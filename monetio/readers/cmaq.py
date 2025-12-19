"""CMAQ File Reader"""

import xarray as xr
from numpy import array, unique
from pandas import to_datetime, to_timedelta

from monetio.grids import get_ioapi_pyresample_area_def, grid_from_dataset
from monetio.models.cmaq_specs import CMAQ_SPECIES

from .base import GriddedReader, register_reader


@register_reader("cmaq")
class CMAQReader(GriddedReader):
    def open_dataset(
        self, files, earth_radius=6370000, convert_to_ppb=True, drop_duplicates=False, **kwargs
    ):
        """
        Reads CMAQ netCDF files.
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
        area_def = get_ioapi_pyresample_area_def(ds, grid)

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
        ds = _get_latlon(ds, area_def)

        # rename dimensions
        ds = ds.rename({"COL": "x", "ROW": "y", "LAY": "z"})

        # convert all gas species to ppbv
        if convert_to_ppb:
            for i in ds.variables:
                if "units" in ds[i].attrs:
                    if "ppmV" in ds[i].attrs["units"]:
                        ds[i] *= 1000.0
                        ds[i].attrs["units"] = "ppbV"

        # convert 'micrograms to \mu g'
        for i in ds.variables:
            if "units" in ds[i].attrs:
                if "micrograms" in ds[i].attrs["units"]:
                    ds[i].attrs["units"] = r"$\mu g m^{-3}$"

        # 3. Harmonize (Standardize names)
        ds = self.harmonize(ds)

        return ds

    def harmonize(self, ds):
        # Placeholder for future harmonization logic
        return ds


def cmaq_preprocess(ds):
    """
    Preprocess function to add lazy diagnostic variables.
    Can be passed to xarray.open_mfdataset.
    """
    ds = add_lazy_derived_vars(ds)
    ds = add_lazy_rh(ds)
    return ds


def _get_times(d, drop_duplicates):
    """
    Vectorized function to parse TFLAG time variables.
    """
    tflag = d["TFLAG"].values
    if tflag.ndim == 2:
        dates = tflag[:, 0]
        times = tflag[:, 1]
    else:  # Assuming 3D with a singleton dimension
        dates = tflag[:, 0, 0]
        times = tflag[:, 0, 1]

    # Vectorized datetime parsing
    dates_pd = to_datetime(dates, format="%Y%j")
    times_pd = to_timedelta(
        (times // 10000), unit="h") + \
        to_timedelta((times % 10000) // 100, unit="m") + \
        to_timedelta((times % 100), unit="s"
    )

    final_times = dates_pd + times_pd

    if drop_duplicates:
        _, index = unique(final_times, return_index=True)
        d = d.isel(TSTEP=index)
        d["TSTEP"] = final_times[index]
    else:
        d["TSTEP"] = final_times

    return d.rename({"TSTEP": "time"})


def _get_latlon(dset, area):
    lon, lat = area.get_lonlats()
    dset["longitude"] = xr.DataArray(lon[::-1, :], dims=["ROW", "COL"])
    dset["latitude"] = xr.DataArray(lat[::-1, :], dims=["ROW", "COL"])
    dset = dset.assign_coords(longitude=dset.longitude, latitude=dset.latitude)
    return dset


def add_lazy_derived_vars(ds):
    """
    Adds lazily-computed diagnostic variables to the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to which variables will be added.

    Returns
    -------
    xarray.Dataset
        The dataset with new diagnostic variables.
    """
    for species, formula in CMAQ_SPECIES.items():
        # If an alternative name is already in the dataset, use it and skip calculation
        alt_names = formula.get("alt_names", [])
        species_added = False
        for alt_name in alt_names:
            if alt_name in ds:
                ds[species] = ds[alt_name]
                species_added = True
                break  # Found one, no need to check others

        if species_added:
            continue  # Move to the next species

        # Find which constituent variables are available in the dataset
        available_vars = [v for v in formula["vars"] if v in ds]
        if not available_vars:
            continue

        # Lazily sum the available variables
        # to_array().sum() is a fast, vectorized way to sum DataArrays
        subset_ds = ds[available_vars]

        if "weights" in formula:
            # Create a DataArray of weights aligned with the variables
            weights = xr.DataArray(
                [formula["weights"][formula["vars"].index(v)] for v in available_vars],
                dims=["variable"],
                coords={"variable": available_vars},
            )
            derived_var = (subset_ds.to_array(dim="variable") * weights).sum("variable")
        else:
            derived_var = subset_ds.to_array(dim="variable").sum("variable")

        # Assign the new variable and its attributes to the dataset
        ds[species] = derived_var.assign_attrs(formula.get("attrs", {}))

    return ds


def add_lazy_rh(d):
    # Placeholder as in original code
    return d
