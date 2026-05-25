"""MADIS (Meteorological Assimilation Data Ingest System) Reader"""

import pandas as pd
import xarray as xr

from .base import PointReader, register_reader
from .sat_utils import update_history


@register_reader("madis")
class MADISReader(PointReader):
    """
    Reader for NOAA MADIS (Meteorological Assimilation Data Ingest System) data.
    """

    def open_dataset(self, files: str | list[str], **kwargs) -> xr.Dataset:
        """
        Reads MADIS NetCDF files.

        Parameters
        ----------
        files : str or list[str]
            File path(s) or glob pattern.
        **kwargs : dict
            Additional arguments passed to PandasDriver.open or to_xarray.

        Returns
        -------
        xr.Dataset
            The MADIS dataset.
        """
        # MADIS files are NetCDF but contain point data.
        # We can use xarray to open them and then convert to the MONETIO point format.
        if isinstance(files, str):
            import glob

            files = sorted(glob.glob(files)) if "*" in files else [files]

        datasets = []
        for f in files:
            ds = xr.open_dataset(f, **kwargs)
            # MADIS files often have 'recNum' as the dimension
            if "recNum" in ds.dims:
                ds = ds.rename({"recNum": "node"})
            datasets.append(ds)

        if len(datasets) > 1:
            # Consolidate
            ds = xr.concat(datasets, dim="node")
        else:
            ds = datasets[0]

        return self.harmonize(ds)

    def harmonize(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Harmonize MADIS dataset.

        Parameters
        ----------
        ds : xr.Dataset
            Input dataset.

        Returns
        -------
        xr.Dataset
            Harmonized dataset.
        """
        # Mapping MADIS variable names to MONET names
        mapping = {
            "latitude": "latitude",
            "longitude": "longitude",
            "observationTime": "time",
            "stationId": "siteid",
            "stationName": "name",
            "elevation": "elevation",
            "temperature": "temperature",
            "dewpoint": "dewpoint",
            "relHumidity": "rel_humidity",
            "windDir": "wind_dir",
            "windSpeed": "wind_speed",
            "altimeter": "altimeter",
            "stationPress": "station_pressure",
            "seaLevelPress": "slp",
            "precip": "precipitation",
        }

        rename_dict = {
            old: new
            for old, new in mapping.items()
            if old in ds.variables and new not in ds.variables
        }
        if rename_dict:
            ds = ds.rename(rename_dict)

        # Handle time if it's in seconds since epoch
        if "time" in ds.variables:
            if ds["time"].attrs.get("units") == "seconds since 1970-01-01 00:00:00.0 +0000":
                ds["time"] = pd.to_datetime(ds["time"].values, unit="s")

        # Set coordinates
        coords = ["time", "siteid", "latitude", "longitude", "elevation"]
        ds = ds.set_coords([c for c in coords if c in ds.variables])

        # Update history
        ds = update_history(ds, "Harmonized MADIS data.")

        return ds
