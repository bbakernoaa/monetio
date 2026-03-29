"""RRFS Reader for AWS Open Data"""

import datetime
from typing import List, Union

import pandas as pd
import xarray as xr

from .base import register_reader
from .gfs import NCEPPDSReader


@register_reader("rrfs")
class RRFSReader(NCEPPDSReader):
    """
    Reader for RRFS (Rapid Refresh Forecast System) on AWS.
    """

    def build_urls(
        self,
        dates: Union[pd.DatetimeIndex, List[datetime.datetime], datetime.datetime, str],
        hour: int = 0,
        lead_time: Union[int, List[int]] = 0,
        product: str = "prslev.3km",
        domain: str = "conus",
        **kwargs,
    ) -> List[str]:
        """
        Build S3 URLs for RRFS data.

        Parameters
        ----------
        dates : Union[pd.DatetimeIndex, List[datetime], datetime, str]
            Dates to retrieve.
        hour : int, optional
            Forecast cycle hour.
        lead_time : Union[int, List[int]], optional
            Forecast lead time(s).
        product : str, optional
            Product string, by default "prslev.3km".
        domain : str, optional
            Domain string (e.g., 'conus', 'ak', 'na', 'pr', 'hi'), by default "conus".

        Returns
        -------
        List[str]
            List of S3 URLs.
        """
        if isinstance(dates, (str, datetime.datetime, pd.Timestamp)):
            dates = pd.DatetimeIndex([pd.to_datetime(dates)])
        else:
            dates = pd.to_datetime(dates)

        if isinstance(lead_time, int):
            lead_times = [lead_time]
        else:
            lead_times = lead_time

        bucket = "noaa-rrfs-pds"
        urls = []
        for d in dates:
            d_str = d.strftime("%Y%m%d")
            h_str = f"{hour:02d}"
            for lt in lead_times:
                lt_str = f"{lt:03d}"
                # s3://noaa-rrfs-pds/rrfs_a/rrfs.20260328/00/rrfs.t00z.prslev.3km.f000.conus.grib2
                # In actual path, maybe rrfs_a/rrfs.YYYYMMDD/HH/rrfs.tHHz.{product}.f{lead_time}.{domain}.grib2
                url = f"s3://{bucket}/rrfs_a/rrfs.{d_str}/{h_str}/rrfs.t{h_str}z.{product}.f{lt_str}.{domain}.grib2"
                urls.append(url)
        return urls

    def open_dataset(
        self,
        files: Union[str, List[str]] = None,
        dates: Union[pd.DatetimeIndex, List[datetime.datetime], datetime.datetime, str] = None,
        hour: int = 0,
        lead_time: Union[int, List[int]] = 0,
        product: str = "prslev.3km",
        domain: str = "conus",
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads RRFS data.
        """
        if files is None:
            if dates is None:
                raise ValueError("Either 'files' or 'dates' must be provided.")
            files = self.build_urls(
                dates, hour=hour, lead_time=lead_time, product=product, domain=domain, **kwargs
            )

        return super().open_dataset(files=files, **kwargs)
