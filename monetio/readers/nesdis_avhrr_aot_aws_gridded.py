"""NESDIS AVHRR AOT AWS Gridded Reader"""

from enum import Enum
from typing import List, Optional, Union

import pandas as pd
import s3fs
import xarray as xr

from .base import GriddedReader, register_reader


class AveragingTime(Enum):
    DAILY = "daily"
    MONTHLY = "monthly"


AOD_BASE_PATH = "noaa-cdr-aerosol-optical-thickness-pds/data/daily"
AOD_FILE_PATTERN = "AOT_AVHRR_*_daily-avg_"


@register_reader("nesdis_avhrr_aot_aws_gridded")
class NESDISAVHRRAOTAWSGriddedReader(GriddedReader):
    """
    Reader for NESDIS AVHRR AOT AWS Gridded data.
    """

    def __init__(self):
        super().__init__()
        self.fs = s3fs.S3FileSystem(anon=True)

    def _create_daily_aod_list(
        self, date_generated: List[pd.Timestamp], warning: bool = False
    ) -> List[str]:
        """
        Creates a list of daily AOD files.
        """
        file_list = []
        for date in date_generated:
            file_date = date.strftime("%Y%m%d")
            year = file_date[:4]
            prod_path = f"{AOD_BASE_PATH}/{year}/"
            file_names = self.fs.glob(f"{prod_path}{AOD_FILE_PATTERN}{file_date}_*.nc")

            if file_names:
                file_list.extend(file_names)
            else:
                msg = f"File does not exist on AWS: {prod_path}{AOD_FILE_PATTERN}{file_date}_*.nc"
                if warning:
                    print(msg)
                    file_list.append(None)
                else:
                    raise ValueError(msg)

        return file_list

    def _create_monthly_aod_list(
        self, date_generated: List[pd.Timestamp], warning: bool = False
    ) -> List[str]:
        """
        Creates a list of monthly AOD files.
        """
        file_list = []
        for date in date_generated:
            file_date = date.strftime("%Y%m%d")
            year = file_date[:4]
            prod_path = f"noaa-cdr-aerosol-optical-thickness-pds/data/monthly/{year}/"
            patt = "AOT_AVHRR_*_daily-avg_"
            file_names = self.fs.glob(f"{prod_path}{patt}{file_date}_*.nc")

            if file_names:
                file_list.extend(file_names)
            else:
                msg = f"File does not exist on AWS: {prod_path}{patt}{file_date}_*.nc"
                if warning:
                    print(msg)
                    file_list.append(None)
                else:
                    raise ValueError(msg)

        return file_list

    def open_dataset(
        self,
        files: Union[str, List[str], None] = None,
        date: Union[str, pd.Timestamp, pd.DatetimeIndex, None] = None,
        averaging_time: Union[AveragingTime, str] = AveragingTime.DAILY,
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads NESDIS AVHRR AOT AWS Gridded data.

        Args:
            files: (Ignored for this reader, uses 'date' instead)
            date: Date(s) to download/read data for.
            averaging_time: 'daily' or 'monthly'.
            **kwargs: Additional arguments passed to xarray.

        Returns:
            xarray.Dataset
        """

        if date is None:
            if files is not None:
                if isinstance(files, str):
                    date = files
                else:
                    raise ValueError("Date is required for NESDIS AVHRR AOT AWS Gridded reader.")
            else:
                raise ValueError("Date is required for NESDIS AVHRR AOT AWS Gridded reader.")

        if isinstance(date, (list, pd.DatetimeIndex)) or (isinstance(date, str) and "," in date):
            return self._open_mfdataset(dates=date, averaging_time=averaging_time)
        else:
            return self._open_dataset(date=date, averaging_time=averaging_time)

    def _open_dataset(
        self,
        date: Union[str, pd.Timestamp],
        averaging_time: Union[AveragingTime, str] = AveragingTime.DAILY,
    ) -> xr.Dataset:
        """Open single dataset."""
        if isinstance(date, str):
            date_generated = [pd.Timestamp(date)]
        else:
            date_generated = [date]

        # Get file list based on averaging time
        if isinstance(averaging_time, str):
            averaging_time = AveragingTime(averaging_time.upper())

        if averaging_time == AveragingTime.MONTHLY:
            file_list = self._create_monthly_aod_list(date_generated)
        else:  # daily
            file_list = self._create_daily_aod_list(date_generated)

        if len(file_list) == 0 or all(f is None for f in file_list):
            raise ValueError(f"Files not available for product and date: {date_generated[0]}")

        # Open and process dataset
        dset = xr.open_dataset(self.fs.open(file_list[0]))

        return dset

    def _open_mfdataset(
        self,
        dates: Union[pd.DatetimeIndex, pd.Timestamp, str],
        averaging_time: Union[AveragingTime, str] = AveragingTime.DAILY,
        error_missing: bool = False,
    ) -> xr.Dataset:
        """Open multiple datasets."""
        # Convert dates to DatetimeIndex
        if isinstance(dates, (str, pd.Timestamp)):
            dates = pd.DatetimeIndex([dates])
        elif not isinstance(dates, pd.DatetimeIndex):
            dates = pd.DatetimeIndex(dates)

        # Get file list based on averaging time
        if isinstance(averaging_time, str):
            averaging_time = AveragingTime(averaging_time.upper())

        if averaging_time == AveragingTime.MONTHLY:
            file_list = self._create_monthly_aod_list(dates, warning=not error_missing)
        else:  # daily
            file_list = self._create_daily_aod_list(dates, warning=not error_missing)

        if len(file_list) == 0 or all(f is None for f in file_list):
            raise ValueError(f"Files not available for product and dates: {dates}")

        aws_files = [self.fs.open(f) for f in file_list if f is not None]

        dset = xr.open_mfdataset(aws_files, concat_dim="time", combine="nested")

        return dset
