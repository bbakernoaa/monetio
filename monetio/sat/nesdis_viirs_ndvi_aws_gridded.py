"""
NOAA Climate Data Record (CDR) and Near Real-Time (NRT) Dataset Access Module

This module provides access to various NOAA satellite-derived environmental datasets:

1. Vegetation Health Index (VHI):
   - Available from both VIIRS (2012-present) and AVHRR (1981-2012) sensors
   - Source: NOAA CDR
   - Resolution: 4km global
   - Frequency: Daily
   - Variables: NDVI, VCI, TCI, VHI
   - AWS Path: noaa-cdr-vegetation-health-pds/

2. Leaf Area Index (LAI) and Fraction of Photosynthetically Active Radiation (FPAR):
   - Available from VIIRS sensor (2012-present)
   - Source: NOAA VIIRS
   - Resolution: 500m global
   - Frequency: Daily
   - Variables: LAI, FPAR
   - AWS Path: noaa-viirs-lai-fpar/

3. Snow Cover Extent:
   - Available from Interactive Multisensor Snow and Ice Mapping System (IMS)
   - Source: NOAA CDR
   - Resolution: 4km Northern Hemisphere
   - Frequency: Daily
   - Variables: Snow Cover, Sea Ice
   - AWS Path: noaa-cdr-snow-cover-extent-ims-nrt/

Data Access:
    All datasets are accessed through AWS S3 buckets in NetCDF format.
    Files are organized by year and contain daily observations.

Usage:
    >>> # Single date access
    >>> dataset = open_dataset("2023-01-01", data_type="vhi", sensor="viirs")

    >>> # Multiple dates
    >>> dates = pd.date_range("2023-01-01", "2023-01-10")
    >>> dataset_multi = open_mfdataset(dates, data_type="snow", sensor="ims")

    >>> # Historical AVHRR data
    >>> dataset_avhrr = open_dataset("2000-01-01", data_type="vhi", sensor="avhrr")

References:
    - VHI: https://www.ncdc.noaa.gov/cdr/terrestrial/vegetation-health
    - LAI/FPAR: https://www.star.nesdis.noaa.gov/jpss/EDRs/products_Vegetation.php
    - Snow Cover: https://www.ncdc.noaa.gov/snow-and-ice/snow-cover

Note:
    This module requires active internet connection and access to AWS S3 buckets.
    Some datasets might have temporal gaps or missing data.
"""
# Standard library imports
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Union
import warnings

# Third-party imports
import pandas as pd
import s3fs
import xarray as xr

# Configuration dictionary for different data products
DATA_CONFIGS = {
    "vhi": {
        "viirs": {
            "path": "noaa-cdr-ndvi-pds/data/",
            "pattern": "VIIRS-Land_*"
        },
        "avhrr": {
            "path": "noaa-cdr-vegetation-health-pds/data/",
            "pattern": "AVHRR-Land_*"
        }
    },
    "lai_fpar": {
        "viirs": {
            "path": "noaa-cdr-leaf-area-index-fapar-pds/data/",
            "pattern": "VIIRS-Land_*"
        },
        'avhrr': {
            "path": "noaa-cdr-leaf-area-index-fapar-pds/data/",
            "pattern": "AVHRR-Land_*"
        }
    },
    "snow": {
        "ims": {
            "path": "noaa-cdr-snow-cover-extent-ims-nrt/",
            "pattern": "snow_cover_extent_*"
        }
    }
}

def validate_inputs(date_generated: List[datetime], data_type: str, sensor: str) -> None:
    """
    Validates input parameters.

    Args:
        date_generated: List of dates to process
        data_type: Type of data product
        sensor: Sensor type

    Raises:
        ValueError: If inputs are invalid
    """
    if data_type not in DATA_CONFIGS:
        raise ValueError(f"Unsupported data type: {data_type}. Available types: {list(DATA_CONFIGS.keys())}")

    if sensor not in DATA_CONFIGS[data_type]:
        raise ValueError(
            f"Unsupported sensor '{sensor}' for data type '{data_type}'. "
            f"Available sensors: {list(DATA_CONFIGS[data_type].keys())}"
        )

@lru_cache(maxsize=128)
def _get_cached_file_list(year: str, prod_path: str, pattern: str, file_date: str) -> List[str]:
    """
    Cached version of file listing to improve performance for repeated requests.
    """
    fs = s3fs.S3FileSystem(anon=True)
    return fs.glob(f"{prod_path}{year}/{pattern}{file_date}_*.nc")

def create_daily_data_list(
    date_generated: List[datetime],
    fs: s3fs.S3FileSystem,
    data_type: str = "vhi",
    sensor: str = "viirs",
    warning: bool = False
) -> Tuple[List[str], int]:
    """
    Creates a list of daily data files and calculates the total size of the files.

    Args:
        date_generated: List of dates to process
        fs: S3 filesystem object
        data_type: Type of data product
        sensor: Sensor type
        warning: Whether to warn instead of raising an error for missing files

    Returns:
        Tuple containing list of file paths and total size
    """
    validate_inputs(date_generated, data_type, sensor)

    file_list = []
    total_size = 0
    config = DATA_CONFIGS[data_type][sensor]

    for date in date_generated:
        file_date = date.strftime("%Y%m%d")
        year = file_date[:4]

        try:
            file_names = _get_cached_file_list(
                year,
                config["path"],
                config["pattern"],
                file_date
            )

            if file_names:
                file_list.extend(file_names)
                total_size += sum(fs.size(f) for f in file_names)
            else:
                raise FileNotFoundError(
                    f"No files found for {data_type} ({sensor}) on {file_date}"
                )

        except Exception as e:
            if warning:
                warnings.warn(str(e))
                file_list.append(None)
            else:
                raise ValueError(str(e))

    return file_list, total_size

def process_timeofday(dataset: xr.Dataset) -> xr.Dataset:
    """
    Process TIMEOFDAY variable in dataset.

    Args:
        dataset: Input xarray dataset

    Returns:
        Processed dataset
    """
    if "TIMEOFDAY" in dataset:
        m = dataset["TIMEOFDAY"].attrs.pop("scale_factor")
        b = dataset["TIMEOFDAY"].attrs.pop("add_offset")
        fv = dataset["TIMEOFDAY"].attrs.pop("_FillValue")

        dataset["TIMEOFDAY"] = dataset["TIMEOFDAY"] * m + b
        dataset["TIMEOFDAY"].attrs.update(units="hours")
        dataset = xr.decode_cf(dataset)

        dataset["TIMEOFDAY"] = dataset["TIMEOFDAY"].where(
            dataset["TIMEOFDAY"] != pd.Timedelta(fv * m + b, unit="hours")
        )
    else:
        dataset = xr.decode_cf(dataset)

    return dataset

def open_dataset(
    date: Union[str, datetime],
    data_type: str = "vhi",
    sensor: str = "viirs"
) -> xr.Dataset:
    """
    Opens a dataset for the given date.

    Args:
        date: Date to process
        data_type: Type of data product
        sensor: Sensor type

    Returns:
        Opened xarray dataset
    """
    date_generated = [pd.Timestamp(date)] if isinstance(date, str) else [date]

    fs = s3fs.S3FileSystem(anon=True)
    file_list, _ = create_daily_data_list(date_generated, fs, data_type, sensor)

    if len(file_list) == 0 or all(f is None for f in file_list):
        raise ValueError(f"Files not available for {data_type} ({sensor}) and date: {date_generated[0]}")

    dset = xr.open_dataset(fs.open(file_list[0]), decode_cf=False)
    return process_timeofday(dset)

def open_mfdataset(
    dates: Union[pd.DatetimeIndex, datetime, str],
    data_type: str = "vhi",
    sensor: str = "viirs",
    error_missing: bool = False
) -> xr.Dataset:
    """
    Opens and combines multiple NetCDF files into a single dataset.

    Args:
        dates: Dates to process
        data_type: Type of data product
        sensor: Sensor type
        error_missing: Whether to raise error on missing files

    Returns:
        Combined xarray dataset
    """
    if isinstance(dates, (str, datetime)):
        dates = pd.DatetimeIndex([dates])
    elif not isinstance(dates, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(dates)

    fs = s3fs.S3FileSystem(anon=True)
    file_list, _ = create_daily_data_list(
        dates,
        fs,
        data_type=data_type,
        sensor=sensor,
        warning=not error_missing
    )

    if len(file_list) == 0 or all(f is None for f in file_list):
        raise ValueError(f"Files not available for {data_type} ({sensor}) and dates: {dates}")

    aws_files = [fs.open(f) for f in file_list if f is not None]
    dset = xr.open_mfdataset(
        aws_files,
        concat_dim="time",
        combine="nested",
        decode_cf=False,
    )

    return process_timeofday(dset)