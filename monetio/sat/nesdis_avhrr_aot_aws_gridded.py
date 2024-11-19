"""
NOAA Climate Data Record (CDR) Aerosol Optical Depth (AOD) Dataset Access Module

This module provides access to NOAA's satellite-derived Aerosol Optical Depth data:

Aerosol Optical Depth (AOD):
    - Source: NOAA CDR AVHRR AOT (Aerosol Optical Thickness)
    - Period: 1981-present
    - Sensor: Advanced Very High Resolution Radiometer (AVHRR)
    - Resolution: 0.1° x 0.1° (approximately 11km at equator)
    - Coverage: Global over ocean
    - Temporal Resolution:
        * Daily averages
        * Monthly averages
    - Key Variables:
        * aot_550: Aerosol Optical Thickness at 550nm
        * number_of_retrievals: Number of valid retrievals
        * quality_flags: Quality assurance flags
    - AWS Path: noaa-cdr-aerosol-optical-thickness-pds/

Dataset Description:
    The AVHRR AOT CDR provides a consistent, long-term record of aerosol optical
    thickness over global oceans. This parameter is crucial for:
    - Climate change studies
    - Atmospheric correction
    - Air quality monitoring
    - Radiative forcing calculations

Data Access:
    Files are stored in NetCDF format on AWS S3, organized by:
    - Daily data: /data/daily/YYYY/
    - Monthly data: /data/monthly/YYYY/

Usage:
    >>> # Single date access (daily)
    >>> dataset = open_dataset("2023-01-01")

    >>> # Monthly data
    >>> dataset = open_dataset("2023-01-01", averaging_time=AveragingTime.MONTHLY)

    >>> # Multiple dates
    >>> dates = pd.date_range("2023-01-01", "2023-01-10")
    >>> dataset_multi = open_mfdataset(dates)

References:
    - Dataset Documentation: https://www.ncdc.noaa.gov/cdr/atmospheric/aerosol-optical-thickness
    - Algorithm Theoretical Basis Document (ATBD):
      https://www.ncdc.noaa.gov/cdr/atmospheric/aerosol-optical-thickness/documentation

Notes:
    - Data is only available over ocean surfaces
    - Quality flags should be consulted for optimal data usage
    - Monthly averages are computed from daily data
"""
from s3fs import S3FileSystem
from enum import Enum
from datetime import datetime
from pathlib import Path
import warnings
import pandas as pd
import s3fs
import xarray as xr

AOD_BASE_PATH = "noaa-cdr-aerosol-optical-thickness-pds/data/daily"
AOD_FILE_PATTERN = "AOT_AVHRR_*_daily-avg_"

class AveragingTime(Enum):
    DAILY = "daily"
    MONTHLY = "monthly"

def create_daily_aod_list(
    date_generated: List[datetime],
    fs: S3FileSystem,
    warning: bool = False
) -> Tuple[List[str], int]:
    """
    Creates a list of daily AOD (Aerosol Optical Depth) files and calculates the total size of the files.

    Parameters:
        date_generated (list): A list of dates for which to check the existence of AOD files.
        fs (FileSystem): The file system object used to check file existence and size.
        warning (bool, optional): If True, warns instead of raising error when file not found. Defaults to False.

    Returns:
        tuple[list[str | None], int]: A tuple containing:
            - List of file paths (str) or None for missing files if warning=True
            - Total size of the files in bytes
    """
    # Loop through observation dates & check for files
    nodd_file_list = []
    nodd_total_size = 0
    for date in date_generated:
        file_date = date.strftime("%Y%m%d")
        year = file_date[:4]
        prod_path = Path(AOD_BASE_PATH) / year
        file_names = fs.glob(str(prod_path / f"{AOD_FILE_PATTERN}{file_date}_*.nc"))
        # If file exists, add path to list and add file size to total
        if file_names:
            nodd_file_list.extend(file_names)
            nodd_total_size += sum(fs.size(f) for f in file_names)
        else:
            msg = f"File does not exist on AWS: {prod_path}/{AOD_FILE_PATTERN}{file_date}_*.nc"
            if warning:
                warnings.warn(msg)
                nodd_file_list.append(None)
            else:
                raise ValueError(msg)

    return nodd_file_list, nodd_total_size

def create_monthly_aod_list(date_generated, fs, warning=False):
    """
    Creates a list of daily AOD (Aerosol Optical Depth) files and calculates the total size of the files.

    Parameters:
        date_generated (list): A list of dates for which to check the existence of AOD files.
        fs (FileSystem): The file system object used to check file existence and size.

    Returns:
        tuple: A tuple containing the list of file paths and the total size of the files.
    """
    # Loop through observation dates & check for files
    nodd_file_list = []
    nodd_total_size = 0
    for date in date_generated:
        file_date = date.strftime("%Y%m%d")
        year = file_date[:4]
        prod_path = "noaa-cdr-aerosol-optical-thickness-pds/data/monthly/" + year + "/"
        patt = "AOT_AVHRR_*_daily-avg_"
        file_names = fs.glob(prod_path + patt + file_date + "_*.nc")
        # If file exists, add path to list and add file size to total
        if file_names:
            nodd_file_list.extend(file_names)
            nodd_total_size += sum(fs.size(f) for f in file_names)
        else:
            msg = "File does not exist on AWS: " + prod_path + patt + file_date + "_*.nc"
            if warning:
                warnings.warn(msg)
                nodd_file_list.append(None)
            else:
                raise ValueError(msg)

    return nodd_file_list, nodd_total_size

def open_dataset(
    date: Union[str, datetime],
    averaging_time: AveragingTime = AveragingTime.DAILY
) -> xr.Dataset:
    """
    Opens a dataset for the given date, satellite, data resolution, and averaging time.

    Parameters:
        date (str or datetime.datetime): The date for which to open the dataset.
        averaging_time (str, optional): The averaging time.
            Valid values are 'daily', or 'monthly'. Defaults to 'daily'.

    Returns:
        xarray.Dataset: The opened dataset.

    Raises:
        ValueError: If the input values are invalid.
    """
    if isinstance(date, str):
        date_generated = [pd.Timestamp(date)]
    else:
        date_generated = [date]

    # Access AWS using anonymous credentials
    fs = s3fs.S3FileSystem(anon=True)

    if averaging_time == AveragingTime.MONTHLY:
        file_list, _ = create_monthly_aod_list(date_generated, fs)
    elif averaging_time == AveragingTime.DAILY:
        file_list, _ = create_daily_aod_list(date_generated, fs)
    else:
        raise ValueError(
            f"Invalid input for 'averaging_time' {averaging_time!r}: "
            "Valid values are 'daily' or 'monthly'"
        )

    if len(file_list) == 0 or all(f is None for f in file_list):
        raise ValueError(f"Files not available for product and date: {date_generated[0]}")

    with fs.open(file_list[0]) as aws_file:
        dset = xr.open_dataset(aws_file)

    return dset

def open_mfdataset(dates, averaging_time: AveragingTime = AveragingTime.DAILY, error_missing=False):
    """
    Opens and combines multiple NetCDF files into a single xarray dataset.

    Parameters:
        dates (pandas.DatetimeIndex): The dates for which to retrieve the data.
        averaging_time (str, optional): The averaging time.
            Valid values are 'daily', 'weekly', or 'monthly'. Defaults to 'daily'.

    Returns:
        xarray.Dataset: The combined dataset containing the data for the specified dates.

    Raises:
        ValueError: If the input parameters are invalid.

    """
    from collections.abc import Iterable

    if isinstance(dates, Iterable) and not isinstance(dates, str):
        dates = pd.DatetimeIndex(dates)
    else:
        dates = pd.DatetimeIndex([dates])

    # Access AWS using anonymous credentials
    fs = s3fs.S3FileSystem(anon=True)

    if averaging_time == AveragingTime.MONTHLY:
        file_list, _ = create_monthly_aod_list(dates, fs, warning=not error_missing)
    elif averaging_time == AveragingTime.DAILY:
        file_list, _ = create_daily_aod_list(dates, fs, warning=not error_missing)
    else:
        raise ValueError(
            f"Invalid input for 'averaging_time' {averaging_time!r}: "
            "Valid values are 'daily' or 'monthly'"
        )

    if len(file_list) == 0 or all(f is None for f in file_list):
        raise ValueError(f"Files not available for product and dates: {dates}")

    aws_files = [fs.open(f) for f in file_list if f is not None]

    with xr.open_mfdataset(aws_files, concat_dim="time", combine="nested") as dset:
        return dset
