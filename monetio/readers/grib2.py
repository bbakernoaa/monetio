"""Generalized GRIB2 Reader using grib2io"""

import datetime
from typing import List, Optional, Union

import xarray as xr

from .base import GriddedReader, register_reader


@register_reader("grib2")
class Grib2Reader(GriddedReader):
    """
    Generalized Reader for GRIB2 files using the grib2io engine.
    """

    def open_dataset(
        self,
        files: Union[str, List[str]],
        engine: str = "grib2io",
        filters: Optional[dict] = None,
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads GRIB2 files using xarray and grib2io.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path, list of paths, or glob pattern.
        engine : str, optional
            The xarray engine to use, by default "grib2io".
        filters : dict, optional
            Filters to pass to the engine (if supported), by default None.
        **kwargs : dict
            Additional arguments passed to xarray.open_mfdataset or the driver.

        Returns
        -------
        xr.Dataset
            The processed GRIB2 dataset.
        """
        if "engine" not in kwargs:
            kwargs["engine"] = engine

        if filters is not None and "backend_kwargs" not in kwargs:
            kwargs["backend_kwargs"] = {"filters": filters}

        # Use the driver to open files
        # XarrayDriver handles S3, multiple files, etc.
        ds = self.driver.open(files, **kwargs)

        # Standardize and Harmonize
        ds = self.harmonize(ds)

        # Update history
        history = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read GRIB2 data using {engine}."
        if "history" in ds.attrs:
            ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
        else:
            ds.attrs["history"] = history

        return ds

    def harmonize(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Harmonize GRIB2 metadata to Aero Protocol standards.

        Parameters
        ----------
        ds : xr.Dataset
            Input GRIB2 dataset.

        Returns
        -------
        xr.Dataset
            Harmonized dataset.
        """
        # 1. Coordinate Renaming (common in GRIB2)
        rename_dict = {
            "latitude": "latitude",
            "longitude": "longitude",
            "lat": "latitude",
            "lon": "longitude",
            "lat_0": "latitude",
            "lon_0": "longitude",
            "time": "time",
            "valid_time": "time",
            "step": "step",
        }

        actual_rename = {}
        for k, v in rename_dict.items():
            if k in ds.variables or k in ds.dims:
                if v in ds.dims and k != v:
                    continue
                actual_rename[k] = v

        if actual_rename:
            ds = ds.rename(actual_rename)

        # 2. Ensure latitude/longitude are coordinates
        coord_vars = [v for v in ["latitude", "longitude", "time"] if v in ds.variables]
        if coord_vars:
            ds = ds.set_coords(coord_vars)

        # 3. Scientific Hygiene: Strip whitespace from string attributes
        for var in ds.variables:
            for attr, val in ds[var].attrs.items():
                if isinstance(val, str):
                    ds[var].attrs[attr] = val.strip()

        return ds
