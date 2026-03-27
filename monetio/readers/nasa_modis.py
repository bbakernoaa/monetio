"""NASA MODIS Reader"""

from typing import Any, List, Optional, Union

import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader
from .sat_utils import standardize_satellite_coords, update_history


@register_reader("nasa_modis")
class NASAMODISReader(GriddedReader):
    """
    Reader for NASA MODIS HDF files.
    """

    def open_dataset(
        self,
        files: Optional[Union[str, List[str]]] = None,
        dates: Optional[Any] = None,
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads NASA MODIS swath data.

        Parameters
        ----------
        files : Union[str, List[str]], optional
            File path(s) or URL(s).
        dates : Any, optional
            Dates to retrieve if files are not provided.
        **kwargs : dict
            Additional arguments passed to XarrayDriver.open.

        Returns
        -------
        xr.Dataset
            The NASA MODIS dataset.
        """
        if "preprocess" not in kwargs:
            kwargs["preprocess"] = nasa_modis_preprocess

        ds = super().open_dataset(files, dates, **kwargs)

        # Update history
        ds = update_history(ds, "Read NASA MODIS data.")

        return ds


def nasa_modis_preprocess(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess NASA MODIS dataset: standardize and assign coordinates.
    """
    from ..grids import get_modis_latlon_from_swath_hv, get_sinu_area_def

    # Standardize dimensions
    ds = standardize_satellite_coords(
        ds, y_dim=["YDim:MOD_Grid_BRDF", "y"], x_dim=["XDim:MOD_Grid_BRDF", "x"]
    )

    # Extract tile info from filename if possible
    # We might need the original filename, but ds might not have it easily accessible here
    # if it's already opened. However, often it's in attributes.
    fname = ds.attrs.get("file_name", "")
    if not fname and "history" in ds.attrs:
        # Sometimes it's in history
        pass

    # If we don't have filename, we might be in trouble for h, v
    # But usually NASA MODIS files have global attributes
    h = ds.attrs.get("HORIZONTALTILENUMBER")
    v = ds.attrs.get("VERTICALTILENUMBER")

    if h is not None and v is not None:
        ds = get_modis_latlon_from_swath_hv(h, v, ds)
        ds.attrs["area"] = get_sinu_area_def(ds)

    # Handle Time
    if "time" not in ds.coords:
        # Try to get time from attributes or filename
        range_start = ds.attrs.get("RANGEBEGINNINGDATE")
        time_start = ds.attrs.get("RANGEBEGINNINGTIME")
        if range_start and time_start:
            ds["time"] = pd.to_datetime(f"{range_start} {time_start}")
            ds = ds.expand_dims("time")

    return ds
