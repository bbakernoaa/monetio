"""JPSS Sounding Readers (CrIS and ATMS)"""

import xarray as xr

from .base import GriddedReader, _scientific_hygiene, register_reader
from .sat_utils import standardize_satellite_coords, update_history


@register_reader("jpss_met")
class JPSSMetReader(GriddedReader):
    """
    Reader for JPSS (Suomi-NPP, NOAA-20, NOAA-21) thermodynamic profiles
    from CrIS (Cross-track Infrared Sounder) and ATMS (Advanced Technology
    Microwave Sounder).
    """

    def open_dataset(self, files: str | list[str], **kwargs) -> xr.Dataset:
        """
        Reads JPSS CrIS or ATMS NetCDF/HDF5 files.

        Parameters
        ----------
        files : str or list[str]
            File path(s) or glob pattern.
        **kwargs : dict
            Additional arguments passed to XarrayDriver.open.

        Returns
        -------
        xr.Dataset
            The JPSS meteorological profiles dataset.
        """
        if "preprocess" not in kwargs:
            kwargs["preprocess"] = jpss_met_preprocess

        if "engine" not in kwargs:
            kwargs["engine"] = "h5netcdf"

        ds = super().open_dataset(files, **kwargs)

        # Update history
        ds = update_history(ds, "Read JPSS meteorological profile data.")

        return ds


def jpss_met_preprocess(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess JPSS CrIS/ATMS dataset.

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
    # JPSS data usually has 'nscan', 'npress', 'nlat', 'nlon' etc.
    ds = standardize_satellite_coords(
        ds,
        lat_name="latitude",
        lon_name="longitude",
        y_dim=["nscan", "rows", "y"],
        x_dim=["nlon", "cols", "x"],
        z_dim=["npress", "lev", "level", "z"],
    )

    # 2. Variable renaming (standard names for thermodynamic variables)
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
    ds = update_history(ds, "Preprocessed JPSS CrIS/ATMS data.")

    return ds
