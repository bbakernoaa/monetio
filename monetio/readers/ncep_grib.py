"""NCEP GRIB Reader"""

from typing import Any, List, Union

import numpy as np
import xarray as xr

from .base import GriddedReader, register_reader
from .sat_utils import update_history


@register_reader("ncep_grib")
class NCEPGribReader(GriddedReader):
    """
    Reader for NCEP GRIB files.
    """

    def open_dataset(self, files: Union[str, List[str]], **kwargs: Any) -> xr.Dataset:
        """
        Reads NCEP GRIB files.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path, list of paths, or glob pattern.
        **kwargs : Any
            Additional arguments passed to xarray.open_mfdataset or the driver.

        Returns
        -------
        xarray.Dataset
            The processed NCEP GRIB dataset.

        Examples
        --------
        >>> from monetio.readers.ncep_grib import NCEPGribReader
        >>> reader = NCEPGribReader()
        >>> ds = reader.open_dataset("gfs.*.grib2", engine="pynio")
        """
        # Ensure we have engine='pynio' if not specified
        # Note: pynio is often used for these files but might be hard to install.
        if "engine" not in kwargs:
            kwargs["engine"] = "pynio"

        # Also supports open_mfdataset logic
        if "concat_dim" not in kwargs:
            kwargs["concat_dim"] = "time"

        if "preprocess" not in kwargs:
            kwargs["preprocess"] = ncep_grib_preprocess

        ds = self.driver.open(files, **kwargs)

        # Update history
        ds = update_history(ds, "Read NCEP GRIB data.")

        return ds


def ncep_grib_preprocess(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess function for a single NCEP GRIB file.
    Converts 1D latitude/longitude to 2D coordinates lazily.

    Parameters
    ----------
    ds : xarray.Dataset
        Input NCEP GRIB dataset.

    Returns
    -------
    xarray.Dataset
        Processed dataset with 'latitude' and 'longitude' coordinates on (y, x) dims.
    """
    # 1. Coordinate Renaming
    if "lat_0" in ds.coords:
        ds = ds.rename({"lat_0": "latitude", "lon_0": "longitude"})

    # 2. Generate 2D Latitude and Longitude lazily
    if "latitude" in ds.coords and "longitude" in ds.coords:
        # Check if they are 1D
        if ds.latitude.ndim == 1 and ds.longitude.ndim == 1:
            lat_dim = ds.latitude.dims[0]
            lon_dim = ds.longitude.dims[0]

            # Save 1D values and rename their dimensions for broadcast
            lon1d = ds.longitude.rename({lon_dim: "x"})
            lat1d = ds.latitude.rename({lat_dim: "y"})

            # Broadcast to 2D
            # xr.broadcast will handle both NumPy and Dask lazily
            lon2d, lat2d = xr.broadcast(lon1d, lat1d)

            # Ensure dimension order is (y, x) to match original meshgrid behavior
            lon2d = lon2d.transpose("y", "x")
            lat2d = lat2d.transpose("y", "x")

            # Replace 1D coords in the dataset with index ranges
            ds = ds.assign_coords(
                **{
                    lat_dim: np.arange(ds.sizes[lat_dim]),
                    lon_dim: np.arange(ds.sizes[lon_dim]),
                }
            )
            # Rename dims to y, x
            ds = ds.rename({lat_dim: "y", lon_dim: "x"})

            # Assign 2D coordinates
            ds = ds.assign_coords(
                longitude=lon2d.assign_attrs(
                    {"long_name": "Longitude", "units": "degree_east", "standard_name": "longitude"}
                ),
                latitude=lat2d.assign_attrs(
                    {"long_name": "Latitude", "units": "degree_north", "standard_name": "latitude"}
                ),
            )

            ds = ds.set_coords(["latitude", "longitude"])

    # 3. Scientific Hygiene: Strip whitespace from string attributes
    for var in ds.variables:
        for attr, val in ds[var].attrs.items():
            if isinstance(val, str):
                ds[var].attrs[attr] = val.strip()

    # Update history
    ds = update_history(ds, "Preprocessed NCEP GRIB data.")

    return ds
