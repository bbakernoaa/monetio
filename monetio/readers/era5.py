"""ERA5 (ECMWF Reanalysis v5) Reader"""

import xarray as xr

from .base import GriddedReader, _scientific_hygiene, register_reader
from .sat_utils import standardize_satellite_coords, update_history


@register_reader("era5")
class ERA5Reader(GriddedReader):
    """
    Reader for ERA5 (ECMWF Reanalysis v5) data.
    """

    def open_dataset(self, files: str | list[str], **kwargs) -> xr.Dataset:
        """
        Reads ERA5 NetCDF files.

        Parameters
        ----------
        files : str or list[str]
            File path(s) or glob pattern.
        **kwargs : dict
            Additional arguments passed to XarrayDriver.open.

        Returns
        -------
        xr.Dataset
            The ERA5 dataset.
        """
        if "preprocess" not in kwargs:
            kwargs["preprocess"] = era5_preprocess

        ds = super().open_dataset(files, **kwargs)

        # Update history
        ds = update_history(ds, "Read ERA5 data.")

        return ds


def era5_preprocess(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess ERA5 dataset: standardize coordinates and metadata.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.

    Returns
    -------
    xr.Dataset
        Processed dataset.
    """
    # 1. Standardize dimensions and coordinates
    # ERA5 uses 'latitude', 'longitude', 'time', 'level'.
    ds = standardize_satellite_coords(
        ds,
        lat_name="latitude",
        lon_name="longitude",
        y_dim=["latitude", "lat"],
        x_dim=["longitude", "lon"],
        z_dim=["level", "lev"],
    )

    # 2. Expand 1D coords to 2D for UGRID/CF compliance in MONETIO if needed
    if "latitude" in ds.coords and ds["latitude"].ndim == 1:
        if "longitude" in ds.coords and ds["longitude"].ndim == 1:
            # Use lazy broadcasting
            lons, lats = xr.broadcast(ds.longitude, ds.latitude)
            # Re-assign as 2D coordinates
            ds = ds.assign_coords(longitude=lons, latitude=lats)

    # 3. Variable renaming to standard names if they exist
    # Common ERA5 variable mapping
    mapping = {
        "u10": "u_wind_10m",
        "v10": "v_wind_10m",
        "t2m": "temperature_2m",
        "d2m": "dewpoint_temperature_2m",
        "sp": "surface_pressure",
        "msl": "mean_sea_level_pressure",
        "tp": "total_precipitation",
        "blh": "pbl_height",
        "u": "u_wind",
        "v": "v_wind",
        "t": "temperature",
        "q": "specific_humidity",
        "r": "relative_humidity",
        "z": "geopotential",
    }
    rename_dict = {
        old: new for old, new in mapping.items() if old in ds.variables and new not in ds.variables
    }
    if rename_dict:
        ds = ds.rename(rename_dict)

    # 4. Scientific Hygiene
    ds = _scientific_hygiene(ds)

    # Update history
    ds = update_history(ds, "Preprocessed ERA5 data.")

    return ds
