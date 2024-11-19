"""
NOAA VIIRS Aerosol Optical Depth (AOD) Dataset Access Module

This module provides access to NOAA's VIIRS-derived Aerosol Optical Depth data:

Data Products:
    1. Daily AOD:
        - Resolution options: 0.05°, 0.10°, 0.25°
        - Coverage: Global over ocean
        - Variables: AOD at 550nm, quality flags
        - Path: noaa-jpss/{satellite}/VIIRS/{resolution}_Degrees_Daily/

    2. Weekly AOD:
        - Fixed resolution: 0.25°
        - Coverage: Global over ocean
        - Variables: Weekly averaged AOD
        - Path: noaa-jpss/{satellite}/VIIRS/0.25_Degrees_Weekly/

    3. Monthly AOD:
        - Fixed resolution: 0.25°
        - Coverage: Global over ocean
        - Variables: Monthly averaged AOD
        - Path: noaa-jpss/{satellite}/VIIRS/0.25_Degrees_Monthly/

Satellites:
    - SNPP: Data available from 2012-01-19 to present
    - NOAA20: Data available from 2018-01-01 to present

References:
    - VIIRS AOD Algorithm: https://www.star.nesdis.noaa.gov/jpss/documents/ATBD/ATBD_EPS_Aerosol_AOD_v3.0.1.pdf
    - Data Access: https://www.avl.class.noaa.gov/saa/products/welcome
"""

from typing import List, Tuple, Union
from datetime import datetime
import warnings
import pandas as pd
import s3fs
import xarray as xr
from enum import Enum
from functools import lru_cache
from pathlib import Path

class AveragingTime(str, Enum):
    """Enumeration of valid averaging time periods."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

class Satellite(str, Enum):
    """Enumeration of valid satellites."""
    SNPP = "SNPP"
    NOAA20 = "NOAA20"

# Configuration dictionary for data products
PRODUCT_CONFIG = {
    AveragingTime.DAILY: {
        "path_template": "noaa-jpss/{satellite}/VIIRS/{satellite}_VIIRS_Aerosol_Optical_Depth_Gridded_Reprocessed/{resolution}_Degrees_Daily/{year}/",
        "file_template": "viirs_eps_{sat_name}_aod_{resolution}_deg_{date}.nc",
        "resolutions": {"0.050", "0.100", "0.250"},
    },
    AveragingTime.WEEKLY: {
        "path_template": "noaa-jpss/{satellite}/VIIRS/{satellite}_VIIRS_Aerosol_Optical_Depth_Gridded_Reprocessed/0.25_Degrees_Weekly/{year}/",
        "resolutions": {"0.250"},
    },
    AveragingTime.MONTHLY: {
        "path_template": "noaa-jpss/{satellite}/VIIRS/{satellite}_VIIRS_Aerosol_Optical_Depth_Gridded_Reprocessed/0.25_Degrees_Monthly/",
        "file_template": "viirs_aod_monthly_{sat_name}_0.250_deg_{date}.nc",
        "resolutions": {"0.250"},
    }
}

@lru_cache(maxsize=128)
def _get_satellite_name(satellite: str) -> str:
    """Get the lowercase satellite name used in file paths."""
    return "npp" if satellite == "SNPP" else "noaa20"

def validate_inputs(satellite: str, data_resolution: str, averaging_time: str) -> None:
    """
    Validate input parameters.

    Args:
        satellite: Satellite name
        data_resolution: Data resolution
        averaging_time: Averaging period

    Raises:
        ValueError: If inputs are invalid
    """
    if satellite not in {s.value for s in Satellite}:
        raise ValueError(f"Invalid satellite: {satellite}. Must be one of {list(Satellite)}")

    if averaging_time not in {t.value for t in AveragingTime}:
        raise ValueError(f"Invalid averaging_time: {averaging_time}. Must be one of {list(AveragingTime)}")

    if data_resolution not in PRODUCT_CONFIG[averaging_time]["resolutions"]:
        raise ValueError(
            f"Invalid resolution {data_resolution} for {averaging_time} data. "
            f"Valid resolutions: {PRODUCT_CONFIG[averaging_time]['resolutions']}"
        )

def create_daily_aod_list(
    data_resolution: str,
    satellite: str,
    date_generated: List[datetime],
    fs: s3fs.S3FileSystem,
    warning: bool = False
) -> Tuple[List[str], int]:
    """
    Creates a list of daily AOD files and calculates their total size.

    Args:
        data_resolution: Data resolution
        satellite: Satellite name
        date_generated: List of dates to process
        fs: S3 filesystem object
        warning: Whether to warn instead of raise errors

    Returns:
        Tuple of (file_list, total_size)
    """
    validate_inputs(satellite, data_resolution, AveragingTime.DAILY)

    file_list = []
    total_size = 0
    sat_name = _get_satellite_name(satellite)
    config = PRODUCT_CONFIG[AveragingTime.DAILY]

    for date in date_generated:
        file_date = date.strftime("%Y%m%d")
        year = file_date[:4]

        file_name = config["file_template"].format(
            sat_name=sat_name,
            resolution=data_resolution,
            date=file_date
        )

        prod_path = config["path_template"].format(
            satellite=satellite,
            resolution=data_resolution[:4],
            year=year
        )

        full_path = prod_path + file_name

        if fs.exists(full_path):
            file_list.extend(fs.ls(full_path))
            total_size += fs.size(full_path)
        else:
            msg = f"File does not exist: {full_path}"
            if warning:
                warnings.warn(msg, stacklevel=2)
                file_list.append(None)
            else:
                raise ValueError(msg)

    return file_list, total_size

def create_monthly_aod_list(
    satellite: str,
    date_generated: List[datetime],
    fs: s3fs.S3FileSystem,
    warning: bool = False
) -> Tuple[List[str], int]:
    """
    Creates a list of monthly AOD files and calculates their total size.

    Args:
        satellite: Satellite name
        date_generated: List of dates to process
        fs: S3 filesystem object
        warning: Whether to warn instead of raise errors

    Returns:
        Tuple of (file_list, total_size)
    """
    validate_inputs(satellite, "0.250", AveragingTime.MONTHLY)

    file_list = []
    total_size = 0
    processed_months = set()
    sat_name = _get_satellite_name(satellite)
    config = PRODUCT_CONFIG[AveragingTime.MONTHLY]

    for date in date_generated:
        year_month = date.strftime("%Y%m")
        if year_month in processed_months:
            continue

        processed_months.add(year_month)
        file_name = config["file_template"].format(
            sat_name=sat_name,
            date=year_month
        )

        prod_path = config["path_template"].format(satellite=satellite)
        full_path = prod_path + file_name

        if fs.exists(full_path):
            file_list.extend(fs.ls(full_path))
            total_size += fs.size(full_path)
        else:
            msg = f"File does not exist: {full_path}"
            if warning:
                warnings.warn(msg, stacklevel=2)
                file_list.append(None)
            else:
                raise ValueError(msg)

    return file_list, total_size

def create_weekly_aod_list(
    satellite: str,
    date_generated: List[datetime],
    fs: s3fs.S3FileSystem,
    warning: bool = False
) -> Tuple[List[str], int]:
    """
    Creates a list of weekly AOD files and calculates their total size.

    Args:
        satellite: Satellite name
        date_generated: List of dates to process
        fs: S3 filesystem object
        warning: Whether to warn instead of raise errors

    Returns:
        Tuple of (file_list, total_size)
    """
    validate_inputs(satellite, "0.250", AveragingTime.WEEKLY)

    file_list = []
    total_size = 0
    config = PRODUCT_CONFIG[AveragingTime.WEEKLY]

    for date in date_generated:
        file_date = date.strftime("%Y%m%d")
        year = file_date[:4]

        prod_path = config["path_template"].format(
            satellite=satellite,
            year=year
        )

        try:
            all_files = fs.ls(prod_path)
            for file in all_files:
                file_name = Path(file).name
                date_range = file_name.split("_")[7].split(".")[0]
                file_start, file_end = date_range.split("-")

                if file_start <= file_date <= file_end and file not in file_list:
                    file_list.append(file)
                    total_size += fs.size(file)
        except Exception as e:
            if warning:
                warnings.warn(str(e), stacklevel=2)
            else:
                raise ValueError(str(e))

    return file_list, total_size

def open_dataset(
    date: Union[str, datetime],
    satellite: str = "SNPP",
    data_resolution: Union[float, str] = 0.1,
    averaging_time: str = "daily"
) -> xr.Dataset:
    """
    Load VIIRS AOD data from AWS for the given parameters.

    Args:
        date: The date for which to open the dataset.
            SNPP has data from 2012-01-19 to present.
            NOAA20 has data from 2018-01-01 to present.
        satellite: The satellite to retrieve data from.
            Valid values are 'SNPP' or 'NOAA20'.
        data_resolution: The data resolution.
            Valid values are '0.050', '0.100', or '0.250'. Defaults to 0.1°.
            Only has effect when averaging_time is 'daily'.
            For 'weekly' and 'monthly' data, resolution is always 0.25.
        averaging_time: The averaging time period.
            Valid values are 'daily', 'weekly', or 'monthly'.

    Returns:
        xarray.Dataset: The opened dataset.

    Raises:
        ValueError: If input parameters are invalid.
    """
    validate_inputs(satellite, str(data_resolution).ljust(5, "0"), averaging_time)

    if isinstance(date, str):
        date_generated = [pd.Timestamp(date)]
    else:
        date_generated = [date]

    fs = s3fs.S3FileSystem(anon=True)

    # Get file list based on averaging time
    if averaging_time == AveragingTime.MONTHLY:
        file_list, _ = create_monthly_aod_list(satellite, date_generated, fs)
    elif averaging_time == AveragingTime.WEEKLY:
        file_list, _ = create_weekly_aod_list(satellite, date_generated, fs)
    else:  # daily
        data_resolution = str(data_resolution).ljust(5, "0")
        file_list, _ = create_daily_aod_list(data_resolution, satellite, date_generated, fs)

    if len(file_list) == 0 or all(f is None for f in file_list):
        raise ValueError(f"Files not available for {averaging_time} data and date: {date_generated[0]}")

    # Open and process dataset
    dset = xr.open_dataset(fs.open(file_list[0]))
    dset = dset.expand_dims(time=date_generated).set_coords(["time"])

    return dset

def open_mfdataset(
    dates: Union[pd.DatetimeIndex, datetime, str],
    satellite: str = "SNPP",
    data_resolution: Union[float, str] = 0.1,
    averaging_time: str = "daily",
    error_missing: bool = False
) -> xr.Dataset:
    """
    Opens and combines multiple NetCDF files into a single dataset.

    Args:
        dates: The dates for which to retrieve the data.
            SNPP has data from 2012-01-19 to present.
            NOAA20 has data from 2018-01-01 to present.
        satellite: The satellite name.
            Valid values are 'SNPP' or 'NOAA20'.
        data_resolution: The data resolution.
            Valid values are '0.050', '0.100', or '0.250'. Defaults to 0.1°.
            Only has effect when averaging_time is 'daily'.
            For 'weekly' and 'monthly' data, resolution is always 0.25.
        averaging_time: The averaging time period.
            Valid values are 'daily', 'weekly', or 'monthly'.
        error_missing: If False (default), skip missing files with warning
            and continue processing. Otherwise, raise an error.

    Returns:
        xarray.Dataset: The combined dataset for specified dates.

    Raises:
        ValueError: If input parameters are invalid.
    """
    # Validate inputs
    validate_inputs(satellite, str(data_resolution).ljust(5, "0"), averaging_time)

    # Convert dates to DatetimeIndex
    if isinstance(dates, (str, datetime)):
        dates = pd.DatetimeIndex([dates])
    elif not isinstance(dates, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(dates)

    fs = s3fs.S3FileSystem(anon=True)

    # Get file list based on averaging time
    if averaging_time == AveragingTime.MONTHLY:
        file_list, _ = create_monthly_aod_list(
            satellite, dates, fs, warning=not error_missing
        )
    elif averaging_time == AveragingTime.WEEKLY:
        file_list, _ = create_weekly_aod_list(
            satellite, dates, fs, warning=not error_missing
        )
    else:  # daily
        data_resolution = str(data_resolution).ljust(5, "0")
        file_list, _ = create_daily_aod_list(
            data_resolution, satellite, dates, fs, warning=not error_missing
        )

    if len(file_list) == 0 or all(f is None for f in file_list):
        raise ValueError(f"Files not available for {averaging_time} data and dates: {dates}")

    if not len(file_list) == len(dates):
        raise ValueError(
            "'dates' and discovered file list are not the same length. "
            f"Check your dates input for {averaging_time} frequency."
        )

    # Process valid files and dates
    dates_good = []
    aws_files = []
    for d, f in zip(dates, file_list):
        if f is not None:
            aws_files.append(fs.open(f))
            dates_good.append(d)

    # Combine datasets
    dset = xr.open_mfdataset(aws_files, concat_dim="time", combine="nested")
    dset["time"] = dates_good

    return dset