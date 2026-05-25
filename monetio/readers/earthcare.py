"""
EarthCARE (Earth Cloud, Aerosol and Radiation Explorer) Reader.
Primarily for ATLID (Atmospheric Lidar) L2 Aerosol Profiles (A-AER).
"""

import xarray as xr

from .base import GriddedReader, register_reader
from .sat_utils import standardize_satellite_coords, update_history


@register_reader("earthcare")
class EarthCAREReader(GriddedReader):
    """
    Reader for EarthCARE L2 data (ATLID, MSI, etc.).
    """

    def open_dataset(
        self,
        files: str | list[str],
        group: str | list[str] | None = None,
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads EarthCARE data.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path(s) or URL(s).
        group : str or list of str, optional
            The NetCDF group(s) to open.
        **kwargs : dict
            Additional arguments passed to XarrayDriver.open.

        Returns
        -------
        xr.Dataset
            The EarthCARE dataset.
        """
        if "engine" not in kwargs:
            kwargs["engine"] = "h5netcdf"

        ds = super().open_dataset(files, **kwargs)

        # Preprocessing
        ds = earthcare_preprocess(ds)

        ds = update_history(ds, "Read EarthCARE L2 data.")

        return ds


def earthcare_preprocess(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess EarthCARE dataset.

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
    # EarthCARE JSG (Joint Standard Grid)
    ds = standardize_satellite_coords(
        ds,
        lat_name="latitude",
        lon_name="longitude",
        y_dim=["profile", "n_profile", "JSG_along_track"],
        z_dim=["range", "n_range", "JSG_vertical"],
    )

    # 2. Handle Time
    if "time" not in ds.coords and "time" not in ds.variables:
        for t_var in ["UTC_time", "time_utc"]:
            if t_var in ds.variables:
                ds = ds.rename({t_var: "time"})
                if ds["time"].ndim == 1:
                    ds = ds.set_coords("time")
                break

    ds = update_history(ds, "Preprocessed EarthCARE data.")

    return ds
