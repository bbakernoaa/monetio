"""
E-PROFILE (European ALC network) Reader
"""

from typing import Any, List, Optional, Union

import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader
from .sat_utils import update_history


@register_reader("eprofile")
class EPROFILEReader(GriddedReader):
    """
    Reader for E-PROFILE (European ALC network) NetCDF data.
    """

    def open_dataset(
        self,
        files: Optional[Union[str, List[str]]] = None,
        dates: Optional[Any] = None,
        **kwargs,
    ) -> xr.Dataset:
        """
        Retrieve and load E-PROFILE data.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path, list of paths, or glob pattern.
        dates : Optional[Union[pd.DatetimeIndex, List, pd.Timestamp, str]], optional
            Dates to retrieve if files are not provided (not yet implemented for E-PROFILE), by default None.
        **kwargs : dict
            Additional arguments passed to xarray.open_mfdataset.

        Returns
        -------
        xr.Dataset
            The loaded E-PROFILE data.
        """
        # Default to combined dimensions if not specified
        kwargs.setdefault("combine", "by_coords")

        # Use XarrayDriver (via GriddedReader) to open via super()
        ds = super().open_dataset(files, dates, preprocess=eprofile_preprocess, **kwargs)

        return ds


def eprofile_preprocess(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess E-PROFILE dataset: standardize coordinates and dimensions.

    Parameters
    ----------
    ds : xr.Dataset
        Input E-PROFILE dataset.

    Returns
    -------
    xr.Dataset
        Preprocessed dataset.
    """
    # 1. Standardize Dimensions and Coordinates
    # Common ALC NetCDF dimensions: 'time', 'range', 'level', 'layer', 'timeDim'
    rename_dims = {}
    if "timeDim" in ds.dims:
        rename_dims["timeDim"] = "time"
    if "range" in ds.dims:
        # Keep range as dim, but we might want a 1D coord for it if it's not there
        pass

    if rename_dims:
        ds = ds.rename(rename_dims)

    # 2. Standardize Variable Names
    rename_vars = {}
    # Attenuated backscatter
    for v in ["beta", "backscatter", "attenuated_backscatter_coefficient"]:
        if v in ds.variables and "attenuated_backscatter" not in ds.variables:
            rename_vars[v] = "attenuated_backscatter"

    # Station Metadata
    if "station_altitude" in ds.variables and "elevation" not in ds.variables:
        rename_vars["station_altitude"] = "elevation"
    if "station_latitude" in ds.variables and "latitude" not in ds.variables:
        rename_vars["station_latitude"] = "latitude"
    if "station_longitude" in ds.variables and "longitude" not in ds.variables:
        rename_vars["station_longitude"] = "longitude"

    if rename_vars:
        ds = ds.rename_vars(rename_vars)

    # 3. Coordinate Handling
    # Ensure latitude and longitude are coordinates
    coord_vars = ["latitude", "longitude", "elevation", "time", "range"]
    actual_coords = [v for v in coord_vars if v in ds.variables]
    if actual_coords:
        ds = ds.set_coords(actual_coords)

    # 4. Standard Units and Attributes
    if "latitude" in ds.coords:
        ds["latitude"].attrs.update({"units": "degrees_north", "standard_name": "latitude"})
    if "longitude" in ds.coords:
        ds["longitude"].attrs.update({"units": "degrees_east", "standard_name": "longitude"})
    if "elevation" in ds.coords:
        ds["elevation"].attrs.update({"units": "m", "standard_name": "height_above_mean_sea_level"})

    # 5. Vertical Coordinate Calculation
    # If we have elevation (station height) and range, calculate altitude (absolute height)
    if "range" in ds.variables and "elevation" in ds.coords:
        if "altitude" not in ds.variables:
            # We assume range is in meters. If it has units km, we should convert.
            range_da = ds["range"]
            if range_da.attrs.get("units") == "km":
                range_da = range_da * 1000.0

            ds["altitude"] = ds["elevation"] + range_da
            ds["altitude"].attrs.update(
                {"units": "m", "standard_name": "altitude", "positive": "up"}
            )
            ds = ds.set_coords("altitude")

    # Update history
    ds = update_history(ds, "Preprocessed E-PROFILE data.")

    return ds
