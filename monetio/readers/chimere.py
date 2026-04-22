"""Chimere Reader"""

from functools import partial
from typing import Any

import xarray as xr

from .base import GriddedReader, _scientific_hygiene, register_reader
from .sat_utils import update_history


@register_reader("chimere")
class ChimereReader(GriddedReader):
    """
    Reader for Chimere model output files.
    """

    def open_dataset(
        self,
        files: str | list[str],
        var_list: list[str] = None,
        surf_only: bool = False,
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Reads Chimere netCDF files.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path, list of paths, or glob pattern.
        var_list : list of str, optional
            List of variable names meant to be kept for the analysis, by default None.
        surf_only : bool, optional
            Whether to only keep surface data (layer 0), by default False.
        **kwargs : Any
            Additional arguments passed to the driver.

        Returns
        -------
        xarray.Dataset
            The processed Chimere dataset.
        """
        if "preprocess" not in kwargs:
            kwargs["preprocess"] = partial(
                chimere_preprocess,
                var_list=var_list,
                surf_only=surf_only,
            )

        if "combine" not in kwargs:
            kwargs["combine"] = "nested"
        if "concat_dim" not in kwargs:
            kwargs["concat_dim"] = "time"

        ds = self.driver.open(files, **kwargs)

        ds = self.harmonize(ds)

        # Update history
        ds = update_history(ds, "Read Chimere data.")

        return ds


def chimere_preprocess(
    ds: xr.Dataset, *, var_list: list[str] = None, surf_only: bool = False
) -> xr.Dataset:
    """
    Preprocess function for a single Chimere file.

    Parameters
    ----------
    ds : xarray.Dataset
        Input Chimere dataset.
    var_list : list of str, optional
        List of variables to keep, by default None.
    surf_only : bool, optional
        Whether to keep only surface data, by default False.

    Returns
    -------
    xarray.Dataset
        Processed dataset.
    """
    if var_list is not None:
        drop_vars = set(ds.data_vars) - set(var_list)
        ds = ds.drop_vars(drop_vars, errors="ignore")

    rename_dict = {
        "nav_lat": "latitude",
        "nav_lon": "longitude",
        "time_counter": "time",
        "bottom_top": "z",
    }
    # Only rename if they exist
    rename_dict = {k: v for k, v in rename_dict.items() if k in ds.variables or k in ds.dims}

    if rename_dict:
        ds = ds.rename(rename_dict)
        # Update history
        ds = update_history(ds, f"Renamed coordinates/dimensions: {rename_dict}.")

    if surf_only and "z" in ds.dims:
        ds = ds.isel(z=[0])
        # Update history
        ds = update_history(ds, "Subsetted to surface layer (z=0).")

    ds = ds.reset_coords()
    coords = [c for c in ["latitude", "longitude", "time"] if c in ds.variables]
    ds = ds.set_coords(coords)

    # Ensure lat/lon have standard attributes
    if "latitude" in ds.coords:
        ds["latitude"].attrs.update({"units": "degrees_north", "standard_name": "latitude"})
    if "longitude" in ds.coords:
        ds["longitude"].attrs.update({"units": "degrees_east", "standard_name": "longitude"})

    # Transpose to standard order if dims exist
    dims = [d for d in ["time", "z", "y", "x"] if d in ds.dims]
    if dims:
        ds = ds.transpose(*dims)

    # Scientific Hygiene
    ds = _scientific_hygiene(ds)

    # Update history
    ds = update_history(ds, "Preprocessed Chimere data.")

    return ds
