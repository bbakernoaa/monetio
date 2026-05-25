"""AMDAR/ACARS (Aircraft Meteorological Data Relay) Reader"""

import pandas as pd
import xarray as xr

from .base import PointReader, register_reader
from .sat_utils import update_history


@register_reader("amdar")
class AMDARReader(PointReader):
    """
    Reader for AMDAR/ACARS (Aircraft Meteorological Data Relay) observations.
    """

    def open_dataset(self, files: str | list[str], **kwargs) -> xr.Dataset:
        """
        Reads AMDAR/ACARS data. Files are typically NetCDF (from MADIS or BUFR-converted).

        Parameters
        ----------
        files : str or list[str]
            File path(s) or glob pattern.
        **kwargs : dict
            Additional arguments passed to Xarray.

        Returns
        -------
        xr.Dataset
            The AMDAR dataset.
        """
        if isinstance(files, str):
            import glob

            files = sorted(glob.glob(files)) if "*" in files else [files]

        datasets = []
        for f in files:
            ds = xr.open_dataset(f, **kwargs)
            # Standardize dimension to node
            if "recNum" in ds.dims:
                ds = ds.rename({"recNum": "node"})
            elif "observation" in ds.dims:
                ds = ds.rename({"observation": "node"})
            datasets.append(ds)

        if len(datasets) > 1:
            ds = xr.concat(datasets, dim="node")
        else:
            ds = datasets[0]

        return self.harmonize(ds)

    def harmonize(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Harmonize AMDAR/ACARS dataset.

        Parameters
        ----------
        ds : xr.Dataset
            Input dataset.

        Returns
        -------
        xr.Dataset
            Harmonized dataset.
        """
        # Mapping for aircraft data
        mapping = {
            "latitude": "latitude",
            "longitude": "longitude",
            "observationTime": "time",
            "altitude": "altitude",
            "pressure": "pressure",
            "temperature": "temperature",
            "windDir": "wind_dir",
            "windSpeed": "wind_speed",
            "tailNumber": "siteid",
            "flightNumber": "flight_number",
            "phase": "flight_phase",
        }

        rename_dict = {
            old: new
            for old, new in mapping.items()
            if old in ds.variables and new not in ds.variables
        }
        if rename_dict:
            ds = ds.rename(rename_dict)

        # Handle time if needed
        if "time" in ds.variables:
            if ds["time"].attrs.get("units") == "seconds since 1970-01-01 00:00:00.0 +0000":
                ds["time"] = pd.to_datetime(ds["time"].values, unit="s")

        # Set coordinates
        coords = ["time", "siteid", "latitude", "longitude", "altitude", "pressure"]
        ds = ds.set_coords([c for c in coords if c in ds.variables])

        # Update history
        ds = update_history(ds, "Harmonized AMDAR/ACARS data.")

        return ds
