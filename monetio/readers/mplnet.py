"""MPLNET Reader"""

from typing import List, Union

import xarray as xr

from .base import GriddedReader, register_reader
from .sat_utils import update_history


@register_reader("mplnet")
class MPLNETReader(GriddedReader):
    """
    Reader for MPLNET (NASA Micro-Pulse Lidar Network) V3 NetCDF data.
    """

    def open_dataset(self, files: Union[str, List[str]], **kwargs) -> xr.Dataset:
        """
        Retrieve and load MPLNET data.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path, list of paths, or glob pattern.
        **kwargs : dict
            Additional arguments passed to xarray.open_mfdataset.

        Returns
        -------
        xr.Dataset
            The loaded MPLNET data.
        """
        # Default to combined dimensions if not specified
        kwargs.setdefault("combine", "by_coords")

        # Use XarrayDriver (via GriddedReader) to open
        ds = super().open_dataset(files, preprocess=mplnet_preprocess, **kwargs)

        return ds


def mplnet_preprocess(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess MPLNET dataset: standardize coordinates and dimensions.

    Parameters
    ----------
    ds : xr.Dataset
        Input MPLNET dataset.

    Returns
    -------
    xr.Dataset
        Preprocessed dataset.
    """
    # 1. Standardize Time
    # MPLNET often has 'time' as double (days since start of year or similar)
    # but modern V3 should be CF compliant and xarray should handle it.
    # If not, we might need custom logic.

    # 2. Rename Dimensions/Coordinates
    rename_vars = {}
    if "surface_altitude" in ds.variables:
        rename_vars["surface_altitude"] = "elevation"

    if rename_vars:
        ds = ds.rename_vars(rename_vars)

    # 3. Unit Conversions
    # elevation (surface_altitude) is in km in MPLNET V3
    if "elevation" in ds.coords or "elevation" in ds.data_vars:
        if ds["elevation"].attrs.get("units") == "km":
            ds["elevation"] = ds["elevation"] * 1000.0
            ds["elevation"].attrs["units"] = "m"

    # 4. Coordinate handling
    # Ensure latitude and longitude are coordinates
    coord_vars = ["latitude", "longitude", "elevation", "time"]
    actual_coords = [v for v in coord_vars if v in ds.variables]
    if actual_coords:
        ds = ds.set_coords(actual_coords)

    # Update history
    ds = update_history(ds, "Preprocessed MPLNET data.")

    return ds
