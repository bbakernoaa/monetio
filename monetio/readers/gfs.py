"""GFS, GEFS, and GDAS Readers for AWS Open Data"""

import datetime
from typing import Any

import pandas as pd

from .base import register_reader
from .ncep_pds import NCEPPDSReader


@register_reader("gfs")
class GFSReader(NCEPPDSReader):
    """
    Reader for GFS (Global Forecast System) on AWS or NOMADS.
    """

    def build_urls(
        self,
        dates: pd.DatetimeIndex | list[datetime.datetime] | datetime.datetime | str,
        hour: int = 0,
        lead_time: int | list[int] = 0,
        product: str = "pgrb2.0p25",
        source: str = "aws",
        **kwargs: Any,
    ) -> list[str]:
        """
        Build URLs for GFS data.
        """
        if isinstance(dates, str | datetime.datetime | pd.Timestamp):
            dates = pd.DatetimeIndex([pd.to_datetime(dates)])
        else:
            dates = pd.to_datetime(dates)

        if isinstance(lead_time, int):
            lead_times = [lead_time]
        else:
            lead_times = lead_time

        urls = []
        for d in dates:
            d_str = d.strftime("%Y%m%d")
            h_str = f"{hour:02d}"
            for lt in lead_times:
                lt_str = f"{lt:03d}"
                if source.lower() == "aws":
                    bucket = "noaa-gfs-bdp-pds"
                    url = (
                        f"s3://{bucket}/gfs.{d_str}/{h_str}/atmos/gfs.t{h_str}z.{product}.f{lt_str}"
                    )
                else:
                    # https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.20250325/00/atmos/gfs.t00z.pgrb2.0p25.f000
                    url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{d_str}/{h_str}/atmos/gfs.t{h_str}z.{product}.f{lt_str}"
                urls.append(url)
        return urls


@register_reader("gefs")
class GEFSReader(NCEPPDSReader):
    """
    Reader for GEFS (Global Ensemble Forecast System) on AWS or NOMADS.
    """

    def build_urls(
        self,
        dates: pd.DatetimeIndex | list[datetime.datetime] | datetime.datetime | str,
        hour: int = 0,
        lead_time: int | list[int] = 0,
        product: str = "geavg.tHHz.pgrb2a.0p50",
        source: str = "aws",
        **kwargs: Any,
    ) -> list[str]:
        """
        Build URLs for GEFS data.
        Note: product here usually specifies the member and resolution.
        Example: 'geavg.tHHz.pgrb2a.0p50' for ensemble mean 0.5 deg.
        """
        if isinstance(dates, str | datetime.datetime | pd.Timestamp):
            dates = pd.DatetimeIndex([pd.to_datetime(dates)])
        else:
            dates = pd.to_datetime(dates)

        if isinstance(lead_time, int):
            lead_times = [lead_time]
        else:
            lead_times = lead_time

        urls = []
        h_str = f"{hour:02d}"
        for d in dates:
            d_str = d.strftime("%Y%m%d")
            for lt in lead_times:
                lt_str = f"{lt:03d}"
                # The product string might have 'tHHz' as a placeholder
                prod = product.replace("tHHz", f"t{h_str}z")
                if source.lower() == "aws":
                    bucket = "noaa-gefs-pds"
                    # s3://noaa-gefs-pds/gefs.20250324/00/atmos/pgrb2ap5/geavg.t00z.pgrb2a.0p50.f000
                    res_dir = "pgrb2ap5" if "0p50" in prod else "pgrb2bp5"
                    url = f"s3://{bucket}/gefs.{d_str}/{h_str}/atmos/{res_dir}/{prod}.f{lt_str}"
                else:
                    # https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod/gefs.20250325/00/atmos/pgrb2ap5/geavg.t00z.pgrb2a.0p50.f000
                    res_dir = "pgrb2ap5" if "0p50" in prod else "pgrb2bp5"
                    url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod/gefs.{d_str}/{h_str}/atmos/{res_dir}/{prod}.f{lt_str}"
                urls.append(url)
        return urls


@register_reader("gdas")
class GDASReader(NCEPPDSReader):
    """
    Reader for GDAS (Global Data Assimilation System) on AWS or NOMADS.
    """

    def build_urls(
        self,
        dates: pd.DatetimeIndex | list[datetime.datetime] | datetime.datetime | str,
        hour: int = 0,
        lead_time: int | list[int] = 0,
        product: str = "pgrb2.0p25",
        source: str = "aws",
        **kwargs: Any,
    ) -> list[str]:
        """
        Build URLs for GDAS data.
        """
        if isinstance(dates, str | datetime.datetime | pd.Timestamp):
            dates = pd.DatetimeIndex([pd.to_datetime(dates)])
        else:
            dates = pd.to_datetime(dates)

        if isinstance(lead_time, int):
            lead_times = [lead_time]
        else:
            lead_times = lead_time

        urls = []
        for d in dates:
            d_str = d.strftime("%Y%m%d")
            h_str = f"{hour:02d}"
            for lt in lead_times:
                lt_str = f"{lt:03d}"
                if source.lower() == "aws":
                    bucket = "noaa-gfs-bdp-pds"
                    url = f"s3://{bucket}/gdas.{d_str}/{h_str}/atmos/gdas.t{h_str}z.{product}.f{lt_str}"
                else:
                    # https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gdas.20250325/00/atmos/gdas.t00z.pgrb2.0p25.f000
                    url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gdas.{d_str}/{h_str}/atmos/gdas.t{h_str}z.{product}.f{lt_str}"
                urls.append(url)
        return urls
