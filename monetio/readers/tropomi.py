"""TROPOMI Reader"""

import datetime
from typing import List, Union

import xarray as xr

from .base import GriddedReader, register_reader
from .sat_utils import standardize_satellite_coords


@register_reader("tropomi")
class TROPOMIReader(GriddedReader):
    """
    Reader for TROPOMI L2 (Sentinel-5P) data.
    """

    def open_dataset(
        self,
        files: Union[str, List[str]],
        group: str = "PRODUCT",
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads TROPOMI data.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path(s) or URL(s).
        group : str, optional
            The NetCDF group to open, by default "PRODUCT".
            Standard TROPOMI L2 files store main data in "PRODUCT".
        **kwargs : dict
            Additional arguments passed to XarrayDriver.open.

        Returns
        -------
        xr.Dataset
            The TROPOMI dataset.
        """
        if "preprocess" not in kwargs:
            kwargs["preprocess"] = tropomi_preprocess

        if "engine" not in kwargs:
            kwargs["engine"] = "h5netcdf"

        # TROPOMI files must be opened group-by-group in xarray
        kwargs["group"] = group

        ds = super().open_dataset(files, **kwargs)

        # Update history
        history = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read TROPOMI data."
        if "history" in ds.attrs:
            ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
        else:
            ds.attrs["history"] = history

        return ds


def tropomi_preprocess(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess TROPOMI dataset: standardize coordinates and handle time.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset from a single file/group.

    Returns
    -------
    xr.Dataset
        Processed dataset.
    """
    # 1. Standardize dimensions and coordinates
    # TROPOMI uses 'scanline' and 'ground_pixel'
    ds = standardize_satellite_coords(ds, lat_name="latitude", lon_name="longitude")

    # 2. Handle Time
    # TROPOMI L2 often has 'time' as a reference time (scalar)
    # and 'delta_time' as milliseconds from reference for each scanline.
    if "time" in ds.coords and "delta_time" in ds.data_vars:
        # scan_time = reference_time + delta_time
        # Use apply_ufunc for laziness
        ref_time = ds.coords["time"]
        delta_time = ds.data_vars["delta_time"]

        # If time is a coordinate but not a dimension (which it usually is in TROPOMI)
        if "y" in delta_time.dims:
            scan_time = ref_time + delta_time.astype("timedelta64[ms]")
            ds = ds.assign_coords(time=scan_time)

    # 3. Scientific Hygiene: Expand dims if 'time' is just a coordinate
    if "time" in ds.coords and "time" not in ds.dims:
        # In TROPOMI, time typically varies with 'y' (scanline)
        # We can rename it to 'time' dimension if it's 1D over y
        if ds.coords["time"].dims == ("y",):
            ds = ds.swap_dims({"y": "time"})

    return ds
