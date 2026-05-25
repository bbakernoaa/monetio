"""MRMS (Multi-Radar Multi-Sensor) Reader"""

import xarray as xr

from .base import GriddedReader, _scientific_hygiene, register_reader
from .sat_utils import standardize_satellite_coords, update_history


@register_reader("mrms")
class MRMSReader(GriddedReader):
    """
    Reader for NOAA MRMS (Multi-Radar Multi-Sensor) gridded precipitation data.
    """

    def open_dataset(self, files: str | list[str], **kwargs) -> xr.Dataset:
        """
        Reads MRMS GRIB2 or NetCDF files.

        Parameters
        ----------
        files : str or list[str]
            File path(s) or glob pattern.
        **kwargs : dict
            Additional arguments passed to XarrayDriver.open.

        Returns
        -------
        xr.Dataset
            The MRMS dataset.
        """
        if "preprocess" not in kwargs:
            kwargs["preprocess"] = mrms_preprocess

        # For GRIB2 files, use the grib2io engine if available
        if isinstance(files, str) and files.endswith((".grib2", ".bin")):
            if "engine" not in kwargs:
                kwargs["engine"] = "grib2io"

        ds = super().open_dataset(files, **kwargs)

        # Update history
        ds = update_history(ds, "Read MRMS data.")

        return ds


def mrms_preprocess(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess MRMS dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.

    Returns
    -------
    xr.Dataset
        Processed dataset.
    """
    # 1. Standardize coordinates
    ds = standardize_satellite_coords(
        ds,
        lat_name="lat",
        lon_name="lon",
        y_dim=["lat", "latitude"],
        x_dim=["lon", "longitude"],
    )

    # 2. Variable renaming (MRMS variables can vary by product)
    mapping = {
        "precip_rate": "precipitation_rate",
        "precip_accum": "precipitation_accumulation",
        "radar_quality_index": "rqi",
    }
    rename_dict = {
        old: new for old, new in mapping.items() if old in ds.variables and new not in ds.variables
    }
    if rename_dict:
        ds = ds.rename(rename_dict)

    # 3. Scientific Hygiene
    ds = _scientific_hygiene(ds)

    # Update history
    ds = update_history(ds, "Preprocessed MRMS data.")

    return ds
