"""NESDIS EPS VIIRS Reader"""

import datetime
from typing import List, Union

import numpy as np
import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader
from .sat_utils import add_time_coord


@register_reader("nesdis_eps_viirs")
class NESDISEPSVIIRSReader(GriddedReader):
    """
    Reader for NESDIS EPS VIIRS (Enterprise Processing System) AOT data.
    """

    def open_dataset(
        self,
        files: Union[str, List[str]] = None,
        dates: Union[pd.DatetimeIndex, List[datetime.datetime], datetime.datetime, str] = None,
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads NESDIS EPS VIIRS data.

        Parameters
        ----------
        files : Union[str, List[str]], optional
            File path(s) or URL(s).
        dates : Union[pd.DatetimeIndex, List[datetime], datetime, str], optional
            Dates to retrieve. If files is None, this is used to build URLs.
        **kwargs : dict
            Additional arguments passed to XarrayDriver.open.

        Returns
        -------
        xr.Dataset
            The NESDIS EPS VIIRS dataset.
        """
        if files is None:
            if dates is None:
                raise ValueError("Either 'files' or 'dates' must be provided.")
            files = self.build_urls(dates)

        if "preprocess" not in kwargs:
            kwargs["preprocess"] = nesdis_eps_viirs_preprocess

        ds = super().open_dataset(files, **kwargs)

        # Standardize naming and attributes
        ds = self.harmonize(ds)

        # Update history
        history = (
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read NESDIS EPS VIIRS data."
        )
        if "history" in ds.attrs:
            ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
        else:
            ds.attrs["history"] = history

        return ds

    def build_urls(
        self, dates: Union[pd.DatetimeIndex, List[datetime.datetime], datetime.datetime, str]
    ) -> List[str]:
        """
        Build FTP URLs for NESDIS EPS VIIRS data based on dates.

        Parameters
        ----------
        dates : Union[pd.DatetimeIndex, List[datetime], datetime, str]
            Dates to retrieve.

        Returns
        -------
        List[str]
            List of FTP URLs.
        """
        if isinstance(dates, (str, datetime.datetime, pd.Timestamp)):
            dates = pd.DatetimeIndex([pd.to_datetime(dates)])
        else:
            dates = pd.to_datetime(dates)

        server = "ftp.star.nesdis.noaa.gov"
        base_dir = "/pub/smcd/VIIRS_Aerosol/npp.viirs.aerosol.data/epsaot550"

        urls = []
        for d in dates:
            year = d.strftime("%Y")
            yyyymmdd = d.strftime("%Y%m%d")
            # Example: ftp://ftp.star.nesdis.noaa.gov/pub/smcd/VIIRS_Aerosol/npp.viirs.aerosol.data/epsaot550/2023/npp_eaot_ip_gridded_0.25_20230101.high.nc
            url = f"ftp://{server}{base_dir}/{year}/npp_eaot_ip_gridded_0.25_{yyyymmdd}.high.nc"
            urls.append(url)
        return urls

    def harmonize(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Harmonize the dataset (placeholder for common transformations).
        """
        return ds


def nesdis_eps_viirs_preprocess(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess NESDIS EPS VIIRS dataset: assign coordinates and rename dimensions.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset from a single file.

    Returns
    -------
    xr.Dataset
        Processed dataset.
    """
    # 1. Identify grid size
    # EPS files typically have nlat=720, nlon=1440
    nlat = ds.sizes.get("nlat", 720)
    nlon = ds.sizes.get("nlon", 1440)

    # 2. Generate coordinates
    lon_min = -179.875
    lon_max = -1.0 * lon_min
    lat_min = -89.875
    lat_max = -1.0 * lat_min
    lons = np.linspace(lon_min, lon_max, nlon)
    lats = np.linspace(lat_max, lat_min, nlat)  # EPS uses descending latitudes (lat_max to lat_min)

    # 3. Rename dimensions
    rename_dict = {}
    if "nlat" in ds.dims:
        rename_dict["nlat"] = "y"
    if "nlon" in ds.dims:
        rename_dict["nlon"] = "x"
    if rename_dict:
        ds = ds.rename(rename_dict)

    # 4. Assign lat/lon
    lon2d, lat2d = np.meshgrid(lons, lats)
    ds = ds.assign_coords(
        latitude=(("y", "x"), lat2d, {"units": "degrees_north", "standard_name": "latitude"}),
        longitude=(("y", "x"), lon2d, {"units": "degrees_east", "standard_name": "longitude"}),
    )

    # 5. Extract time from global attributes
    ds = add_time_coord(ds, time_attr="time_coverage_start")
    if "time" not in ds.coords:
        ds = add_time_coord(ds, time_attr="DATE")

    # 6. Final cleaning and standardization
    if "aot_ip_out" in ds.data_vars:
        ds = ds.rename({"aot_ip_out": "aod_550"})
        ds["aod_550"] = ds["aod_550"].where(ds["aod_550"] > 0)
        ds["aod_550"].attrs.update(
            {
                "long_name": "Aerosol Optical Thickness at 550nm",
                "units": "1",
                "standard_name": "atmosphere_optical_thickness_due_to_ambient_aerosol",
            }
        )

    return ds
