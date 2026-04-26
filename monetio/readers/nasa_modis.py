"""NASA MODIS Reader"""

import pandas as pd
import xarray as xr

from .base import GriddedReader, _scientific_hygiene, register_reader
from .sat_utils import standardize_satellite_coords, update_history


@register_reader("nasa_modis")
class NASAMODISReader(GriddedReader):
    """
    Reader for NASA MODIS HDF files.
    """

    def open_dataset(self, files: str | list[str], **kwargs) -> xr.Dataset:
        """
        Reads NASA MODIS swath data.

        Parameters
        ----------
        files : str | list[str]
            File path, list of paths, or glob pattern.
        **kwargs : Any
            Additional arguments passed to the Xarray driver.

        Returns
        -------
        xr.Dataset
            The processed NASA MODIS dataset.

        Examples
        --------
        >>> from monetio.readers.nasa_modis import NASAMODISReader
        >>> reader = NASAMODISReader()
        >>> ds = reader.open_dataset("MOD43A4.A2023001.h10v05.006.2023010123456.hdf")
        """
        if "preprocess" not in kwargs:
            kwargs["preprocess"] = nasa_modis_preprocess

        ds = super().open_dataset(files, **kwargs)

        # Update history
        ds = update_history(ds, "Read NASA MODIS data.")

        return ds


def nasa_modis_preprocess(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess NASA MODIS dataset: standardize coordinates, handle time, and hygiene.

    Parameters
    ----------
    ds : xr.Dataset
        The raw NASA MODIS dataset.

    Returns
    -------
    xr.Dataset
        The preprocessed NASA MODIS dataset.

    Examples
    --------
    >>> ds = nasa_modis_preprocess(ds)
    """
    from ..grids import get_modis_latlon_from_swath_hv, get_sinu_area_def

    # Standardize dimensions
    ds = standardize_satellite_coords(
        ds, y_dim=["YDim:MOD_Grid_BRDF", "y"], x_dim=["XDim:MOD_Grid_BRDF", "x"]
    )
    ds = update_history(ds, "Standardized satellite coordinates.")

    # Extract tile info from attributes
    h = ds.attrs.get("HORIZONTALTILENUMBER")
    v = ds.attrs.get("VERTICALTILENUMBER")

    if h is not None and v is not None:
        ds = get_modis_latlon_from_swath_hv(h, v, ds)
        ds.attrs["area"] = get_sinu_area_def(ds)
        ds = update_history(ds, f"Assigned coordinates for tile h{h}v{v}.")

    # Handle Time
    if "time" not in ds.coords:
        # Try to get time from attributes
        range_start = ds.attrs.get("RANGEBEGINNINGDATE")
        time_start = ds.attrs.get("RANGEBEGINNINGTIME")
        if range_start and time_start:
            # We use xarray-native assignment to maintain laziness if possible,
            # though these are usually scalar attributes.
            dt = pd.to_datetime(f"{range_start} {time_start}")
            ds = ds.assign_coords(time=dt).expand_dims("time")
            ds = update_history(ds, f"Assigned time coordinate from attributes: {dt}.")

    # Scientific Hygiene
    ds = _scientific_hygiene(ds)

    # Update history
    ds = update_history(ds, "Preprocessed NASA MODIS data.")

    return ds
