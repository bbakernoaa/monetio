"""MOPITT Reader"""

import datetime
from typing import List, Union

import numpy as np
import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader
from .sat_utils import standardize_satellite_coords, update_history


@register_reader("mopitt")
class MOPITTReader(GriddedReader):
    """
    Reader for MOPITT (Measurements Of Pollution In The Troposphere) L3 data.
    """

    def open_dataset(
        self,
        files: Union[str, List[str]],
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads MOPITT data.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path(s) or URL(s).
        **kwargs : dict
            Additional arguments passed to XarrayDriver.open.

        Returns
        -------
        xr.Dataset
            The MOPITT dataset.
        """
        if "preprocess" not in kwargs:
            kwargs["preprocess"] = mopitt_preprocess

        if "engine" not in kwargs:
            kwargs["engine"] = "h5netcdf"

        ds = super().open_dataset(files, **kwargs)

        # Update history
        ds = update_history(ds, "Read MOPITT L3 data.")

        return ds


def mopitt_preprocess(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess MOPITT dataset: standardize coordinates and handle metadata.
    """
    # MOPITT L3 HDF5 structure: /HDFEOS/GRIDS/MOP03/Data Fields/
    # If opened with group selection or root

    mapping = {
        "HDFEOS/GRIDS/MOP03/Data Fields/Latitude": "latitude",
        "HDFEOS/GRIDS/MOP03/Data Fields/Longitude": "longitude",
        "HDFEOS/GRIDS/MOP03/Data Fields/RetrievedCOTotalColumnDay": "co_column",
        # etc.
    }

    for old, new in mapping.items():
        if old in ds.variables:
            ds = ds.rename({old: new})

    # Standardize
    ds = standardize_satellite_coords(ds)

    # Handle time from attributes if missing
    if "time" not in ds.coords:
        # Check for StartTime in attributes
        start_time = ds.attrs.get("StartTime")
        if start_time is not None:
            if isinstance(start_time, (list, np.ndarray)):
                start_time = start_time[0]
            # MOPITT uses seconds since 1993-01-01
            dt = datetime.datetime(1993, 1, 1) + datetime.timedelta(seconds=float(start_time))
            ds["time"] = [pd.to_datetime(dt)]
            ds = ds.expand_dims("time")

    return ds
