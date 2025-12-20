"""NESDIS EPS VIIRS AOD NRT Reader"""

from typing import List, Union

import numpy as np
import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader
from .drivers import XarrayDriver

# Configuration
SERVER = "ftp.star.nesdis.noaa.gov"
BASE_DIR = "/pub/smcd/VIIRS_Aerosol/npp.viirs.aerosol.data/epsaot550/"


@register_reader("nesdis_eps_viirs_aod_nrt")
class NESDISEPSVIIRSAODNRTReader(GriddedReader):
    """
    Reader for NESDIS EPS VIIRS AOD NRT data.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.driver = XarrayDriver()

    def open_dataset(
        self,
        files: Union[str, List[str], None] = None,
        date: Union[str, pd.Timestamp, pd.DatetimeIndex, None] = None,
        satellite: str = "NOAA20",
        data_resolution: float = 0.1,
        daily: bool = True,
        error_missing: bool = False,
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads NESDIS EPS VIIRS AOD NRT data.

        Args:
            files: (Ignored for this reader, uses 'date' instead)
            date: Date(s) to download/read data for.
            satellite: 'NOAA20' or 'SNPP'.
            data_resolution: 0.1 or 0.25.
            daily: True for daily data, False for monthly.
            error_missing: If True, raise error if files are missing.
            **kwargs: Additional arguments passed to xarray.

        Returns:
            xarray.Dataset
        """
        if date is None:
            if files is not None:
                if isinstance(files, str):
                    date = files
                else:
                    raise ValueError("Date is required for NESDIS EPS VIIRS AOD NRT reader.")
            else:
                raise ValueError("Date is required for NESDIS EPS VIIRS AOD NRT reader.")

        if isinstance(date, (list, pd.DatetimeIndex)) or (isinstance(date, str) and "," in date):
            return self._open_mfdataset(
                dates=date,
                satellite=satellite,
                data_resolution=data_resolution,
                daily=daily,
                error_missing=error_missing,
                **kwargs,
            )
        else:
            return self._open_dataset(
                date=date,
                satellite=satellite,
                data_resolution=data_resolution,
                daily=daily,
                **kwargs,
            )

    def _build_urls(self, dates, *, daily=True, data_resolution=0.1, satellite="NOAA20"):
        """Construct URLs for downloading NEPS data."""
        if isinstance(dates, pd.DatetimeIndex):
            dates = dates
        else:
            dates = pd.DatetimeIndex([dates])

        if daily:
            dates = dates.floor("D").unique()
        else:  # monthly
            dates = dates.to_period("M").to_timestamp().unique()

        if data_resolution != 0.25 and not daily:
            print(
                "Monthly data is only available at 0.25 deg resolution, "
                f"got 'data_resolution' {data_resolution!r}"
            )

        sat_dirname = satellite.lower()
        if satellite.upper() == "SNPP":
            sat = "npp" if daily else "snpp"
        elif satellite.upper() == "NOAA20":
            sat = "noaa20"
        res = str(data_resolution).ljust(5, "0")
        aod_dirname = "aod/eps" if daily else "aod_monthly"

        urls = []
        fnames = []

        print("Building VIIRS URLs...")
        base_url = (
            "https://www.star.nesdis.noaa.gov/pub/smcd/VIIRS_Aerosol/viirs_aerosol_gridded_data/"
            f"{sat_dirname}/{aod_dirname}/"
        )

        for date in dates:
            if daily:
                fname = "{}/viirs_eps_{}_aod_{}_deg_{}_nrt.nc".format(
                    date.strftime("%Y"),
                    sat,
                    res,
                    date.strftime("%Y%m%d"),
                )
            else:
                fname = "viirs_aod_monthly_{}_{}_deg_{}_nrt.nc".format(
                    sat,
                    res,
                    date.strftime("%Y%m"),
                )
            url = base_url + fname
            urls.append(url)
            fnames.append(fname)

        return urls, fnames

    def _get_latlons(self, nlat, nlon):
        """Get latitude and longitude grids."""
        lon_min = -179.875
        lon_max = -1 * lon_min
        lat_min = -89.875
        lat_max = -1.0 * lat_min
        lons = np.linspace(lon_min, lon_max, nlon)
        lats = np.linspace(lat_max, lat_min, nlat)
        lon, lat = np.meshgrid(lons, lats)
        return lon, lat

    def _open_dataset(
        self, date, *, satellite="NOAA20", data_resolution=0.1, daily=True, **kwargs
    ):
        """Open single dataset."""
        if not isinstance(date, pd.Timestamp):
            d = pd.to_datetime(date)
        else:
            d = date

        if satellite.lower() not in ("noaa20", "snpp"):
            raise ValueError(
                f"Invalid input for 'satellite' {satellite!r}: "
                "Valid values are 'NOAA20' or 'SNPP'"
            )

        if data_resolution not in {0.1, 0.25}:
            raise ValueError(
                f"Invalid input for 'data_resolution' {data_resolution!r}: "
                "Valid values are 0.1 or 0.25"
            )

        urls, _ = self._build_urls(
            d, satellite=satellite, data_resolution=data_resolution, daily=daily
        )

        dset = self.driver.open(urls[0], **kwargs)
        dset = dset.expand_dims(time=[d]).set_coords(["time"])

        return dset

    def open_mfdataset(
        self,
        dates: Union[pd.DatetimeIndex, List[str]],
        files: Union[str, List[str], None] = None,
        satellite: str = "NOAA20",
        data_resolution: float = 0.1,
        daily: bool = True,
        error_missing: bool = False,
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads multiple NESDIS EPS VIIRS AOD NRT data.

        Args:
            dates: Date(s) to download/read data for.
            files: (Ignored for this reader, uses 'date' instead)
            satellite: 'NOAA20' or 'SNPP'.
            data_resolution: 0.1 or 0.25.
            daily: True for daily data, False for monthly.
            error_missing: If True, raise error if files are missing.
            **kwargs: Additional arguments passed to xarray.

        Returns:
            xarray.Dataset
        """
        return self._open_mfdataset(
            dates=dates,
            satellite=satellite,
            data_resolution=data_resolution,
            daily=daily,
            error_missing=error_missing,
            **kwargs,
        )

    def _open_mfdataset(
        self,
        dates,
        satellite="NOAA20",
        data_resolution=0.1,
        daily=True,
        error_missing=False,
        **kwargs,
    ):
        """Open multiple datasets."""
        import warnings
        from collections.abc import Iterable

        if isinstance(dates, Iterable) and not isinstance(dates, str):
            dates = pd.DatetimeIndex(dates)
        else:
            dates = pd.DatetimeIndex([dates])

        if satellite.lower() not in ("noaa20", "snpp"):
            raise ValueError(
                f"Invalid input for 'satellite' {satellite!r}: "
                "Valid values are 'NOAA20' or 'SNPP'"
            )

        if data_resolution not in {0.1, 0.25}:
            raise ValueError(
                f"Invalid input for 'data_resolution' {data_resolution!r}: "
                "Valid values are 0.1 or 0.25"
            )

        urls, _ = self._build_urls(
            dates, satellite=satellite, data_resolution=data_resolution, daily=daily
        )

        dsets = []
        for url, date in zip(urls, dates):
            try:
                ds = self.driver.open(url, **kwargs)
                ds = ds.expand_dims(time=[date]).set_coords(["time"])
                dsets.append(ds)
            except Exception as e:
                msg = f"Failed to access file on NESDIS FTP server: {url}. Error: {e}"
                if error_missing:
                    raise RuntimeError(msg)
                else:
                    warnings.warn(msg)

        if len(dsets) == 0:
            raise ValueError(f"Files not available for product and dates: {dates}")

        dset = xr.concat(dsets, dim="time")

        return dset
