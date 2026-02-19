"""Chimere Reader"""

import datetime
from functools import partial
from typing import List, Union

import xarray as xr

from .base import GriddedReader, register_reader


@register_reader("chimere")
class ChimereReader(GriddedReader):
    """
    Reader for Chimere model output files.
    """

    def open_dataset(
        self,
        files: Union[str, List[str]],
        var_list: List[str] = None,
        surf_only: bool = False,
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads Chimere netCDF files.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path, list of paths, or glob pattern.
        var_list : List[str], optional
            List of variable names meant to be kept for the analysis, by default None.
        surf_only : bool, optional
            Whether to only keep surface data (layer 0), by default False.
        **kwargs : dict
            Additional arguments passed to the driver.

        Returns
        -------
        xr.Dataset
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
        history = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read Chimere data."
        if "history" in ds.attrs:
            ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
        else:
            ds.attrs["history"] = history

        return ds


def chimere_preprocess(
    ds: xr.Dataset, *, var_list: List[str] = None, surf_only: bool = False
) -> xr.Dataset:
    """
    Preprocess function for a single Chimere file.

    Parameters
    ----------
    ds : xr.Dataset
        Input Chimere dataset.
    var_list : List[str], optional
        List of variables to keep, by default None.
    surf_only : bool, optional
        Whether to keep only surface data, by default False.

    Returns
    -------
    xr.Dataset
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

    ds = ds.rename(rename_dict)

    if surf_only and "z" in ds.dims:
        ds = ds.isel(z=0).expand_dims("z", axis=1)

    ds = ds.reset_coords()
    coords = [c for c in ["latitude", "longitude", "time"] if c in ds.variables]
    ds = ds.set_coords(coords)

    # Scientific Hygiene: Strip whitespace from all string attributes
    for var in ds.variables:
        for attr, val in ds[var].attrs.items():
            if isinstance(val, str):
                ds[var].attrs[attr] = val.strip()

    # Update history
    history = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Preprocessed Chimere data."
    if "history" in ds.attrs:
        ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
    else:
        ds.attrs["history"] = history

    return ds
