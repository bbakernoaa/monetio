"""NESDIS EPS VIIRS AOD NRT Reader"""

from typing import List, Union

import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader
from .drivers import XarrayDriver


@register_reader("nesdis_eps_viirs_aod_nrt")
class NESDISEPSVIIRSAODNRTReader(GriddedReader):
    """
    Reader for NESDIS EPS VIIRS AOD NRT data.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.driver = XarrayDriver()

    def _build_urls(self, dates, *, daily=True, data_resolution=0.1, satellite="NOAA20"):
        """Construct URLs for downloading NEPS data."""
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.to_datetime(dates if isinstance(dates, list) else [dates])

        if daily:
            dates = dates.floor("D").unique()
        else:
            dates = dates.to_period("M").to_timestamp().unique()

        sat_dirname = satellite.lower()
        sat = "npp" if satellite.upper() == "SNPP" else "noaa20"
        res = f"{data_resolution:.3f}"
        aod_dirname = "aod/eps" if daily else "aod_monthly"

        base_url = (
            f"https://www.star.nesdis.noaa.gov/pub/smcd/VIIRS_Aerosol/viirs_aerosol_gridded_data/"
            f"{sat_dirname}/{aod_dirname}/"
        )

        urls = []
        for date in dates:
            year = date.strftime("%Y")
            date_str = date.strftime("%Y%m%d") if daily else date.strftime("%Y%m")
            fname = (
                f"{year}/viirs_eps_{sat}_aod_{res}_deg_{date_str}_nrt.nc"
                if daily
                else f"viirs_aod_monthly_{sat}_{res}_deg_{date_str}_nrt.nc"
            )
            urls.append(base_url + fname)

        return urls

    def open_dataset(
        self,
        *args,
        files: Union[str, List[str], None] = None,
        date: Union[str, pd.Timestamp, pd.DatetimeIndex, None] = None,
        satellite: str = "NOAA20",
        data_resolution: float = 0.1,
        daily: bool = True,
        error_missing: bool = False,
        **kwargs,
    ) -> xr.Dataset:
        if date is None:
            if "dates" in kwargs:
                date = kwargs.pop("dates")
            elif args:
                date = args[0]
            else:
                raise ValueError("Date is required for NESDIS EPS VIIRS AOD NRT reader.")

        if isinstance(date, (str, pd.Timestamp)):
            dates = pd.DatetimeIndex([date])
        elif not isinstance(date, pd.DatetimeIndex):
            dates = pd.DatetimeIndex(date)
        else:
            dates = date
        urls = self._build_urls(
            dates,
            satellite=satellite,
            data_resolution=data_resolution,
            daily=daily,
            **kwargs,
        )

        def preprocess(ds):
            fname = ds.encoding["source"].split("/")[-1]
            ds.attrs["dataset_name"] = fname
            return ds

        try:
            if len(urls) > 1:
                ds = self.driver.open(
                    urls, concat_dim="time", combine="nested", preprocess=preprocess, **kwargs
                )
                return ds.assign_coords(time=dates)
            else:
                ds = self.driver.open(urls[0], **kwargs)
                ds = preprocess(ds)
                return ds.expand_dims(time=dates)
        except Exception as e:
            if error_missing:
                raise
            else:
                import warnings

                warnings.warn(f"Failed to access file on NESDIS FTP server: {e}")
                return xr.Dataset()
