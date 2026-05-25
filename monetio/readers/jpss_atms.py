"""JPSS ATMS (Advanced Technology Microwave Sounder) Reader"""

import xarray as xr

from .base import GriddedReader, _scientific_hygiene, register_reader
from .sat_utils import standardize_satellite_coords, update_history


@register_reader("jpss_atms")
class JPSSATMSReader(GriddedReader):
    """
    Reader for JPSS (Suomi-NPP, NOAA-20, NOAA-21) ATMS (Advanced Technology
    Microwave Sounder) thermodynamic profiles.
    """

    def open_dataset(self, files: str | list[str], **kwargs) -> xr.Dataset:
        """
        Reads JPSS ATMS NetCDF/HDF5 files.

        Parameters
        ----------
        files : str or list[str]
            File path(s) or glob pattern.
        **kwargs : dict
            Additional arguments passed to XarrayDriver.open.

        Returns
        -------
        xr.Dataset
            The JPSS ATMS meteorological profiles dataset.
        """
        if "preprocess" not in kwargs:
            kwargs["preprocess"] = jpss_atms_preprocess

        if "engine" not in kwargs:
            kwargs["engine"] = "h5netcdf"

        ds = super().open_dataset(files, **kwargs)

        # Update history
        ds = update_history(ds, "Read JPSS ATMS meteorological profile data.")

        return ds


def jpss_atms_preprocess(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess JPSS ATMS dataset.

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
        lat_name="latitude",
        lon_name="longitude",
        y_dim=["nscan", "rows", "y"],
        x_dim=["nlon", "cols", "x"],
        z_dim=["npress", "lev", "level", "z"],
    )

    # 2. Variable renaming
    mapping = {
        "temperature": "temperature",
        "water_vapor": "specific_humidity",
        "pressure": "pressure",
        "h2o_mixing_ratio": "water_vapor_mixing_ratio",
        "air_temp": "temperature",
    }
    rename_dict = {
        old: new for old, new in mapping.items() if old in ds.variables and new not in ds.variables
    }
    if rename_dict:
        ds = ds.rename(rename_dict)

    # 3. Scientific Hygiene
    ds = _scientific_hygiene(ds)

    # Update history
    ds = update_history(ds, "Preprocessed JPSS ATMS data.")

    return ds
