"""NCEP GRIB Reader"""

from typing import Any

import numpy as np
import xarray as xr

from .base import GriddedReader, _scientific_hygiene, register_reader
from .sat_utils import update_history


@register_reader("ncep_grib")
class NCEPGribReader(GriddedReader):
    """
    Reader for NCEP GRIB files.
    """

    def open_dataset(self, files: str | list[str], **kwargs: Any) -> xr.Dataset:
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

    Converts 1D latitude/longitude to 2D coordinates lazily and applies
    scientific hygiene.

    Parameters
    ----------
    ds : xarray.Dataset
        Input NCEP GRIB dataset.

    Returns
    -------
    xarray.Dataset
        Processed dataset with 'latitude' and 'longitude' coordinates on (y, x) dims.

    Examples
    --------
    >>> ds = ncep_grib_preprocess(ds)
    """
    # 1. Coordinate Renaming & Promotion
    # Some backends might have them as variables but not coords
    rename_dict = {}
    if "lat_0" in ds.variables:
        rename_dict["lat_0"] = "latitude"
    if "lon_0" in ds.variables:
        rename_dict["lon_0"] = "longitude"

    if rename_dict:
        ds = ds.rename(rename_dict)
        # Promote renamed variables to coordinates
        to_coord = [v for v in rename_dict.values() if v in ds.variables and v not in ds.coords]
        if to_coord:
            ds = ds.set_coords(to_coord)

    # 2. Generate 2D Latitude and Longitude lazily
    if "latitude" in ds.coords and "longitude" in ds.coords:
        # Check if they are 1D
        if ds.latitude.ndim == 1 and ds.longitude.ndim == 1:
            lat_dim = ds.latitude.dims[0]
            lon_dim = ds.longitude.dims[0]

            # Extract data and attributes to create new DataArrays for broadcast.
            # This avoids alignment issues when re-assigning to the dataset.
            lon_data = ds.longitude.data
            lat_data = ds.latitude.data

            # If the dataset is chunked but coordinates are not, wrap them in dask
            # to maintain laziness throughout the broadcast.
            if hasattr(ds, "chunks") and ds.chunks:
                try:
                    import dask.array as da

                    if not hasattr(lon_data, "dask"):
                        lon_data = da.from_array(lon_data, chunks=ds.chunks.get(lon_dim, -1))
                    if not hasattr(lat_data, "dask"):
                        lat_data = da.from_array(lat_data, chunks=ds.chunks.get(lat_dim, -1))
                except ImportError:
                    pass

            lon1d = xr.DataArray(lon_data, dims="x", attrs=ds.longitude.attrs)
            lat1d = xr.DataArray(lat_data, dims="y", attrs=ds.latitude.attrs)

            # Broadcast to 2D
            # xr.broadcast will handle both NumPy and Dask lazily
            lon2d, lat2d = xr.broadcast(lon1d, lat1d)

            # Ensure dimension order is (y, x) to match original meshgrid behavior
            # We use transpose to ensure the order is correct for the new dimensions
            if "y" in lon2d.dims and "x" in lon2d.dims:
                lon2d = lon2d.transpose("y", "x")
                lat2d = lat2d.transpose("y", "x")

            # Replace 1D coords in the dataset with index ranges to avoid dim/coord conflict
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
            ds = update_history(ds, "Generated 2D latitude/longitude coordinates lazily.")

    # 3. Scientific Hygiene
    ds = _scientific_hygiene(ds)

    # Update history
    ds = update_history(ds, "Preprocessed NCEP GRIB data.")

    return ds
