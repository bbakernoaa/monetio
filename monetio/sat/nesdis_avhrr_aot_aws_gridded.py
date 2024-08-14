def create_daily_aod_list(date_generated, fs, warning=False):
    """
    Creates a list of daily AOD (Aerosol Optical Depth) files and calculates the total size of the files.

    Parameters:
        date_generated (list): A list of dates for which to check the existence of AOD files.
        fs (FileSystem): The file system object used to check file existence and size.

    Returns:
        tuple: A tuple containing the list of file paths and the total size of the files.
    """
    import warnings

    # Loop through observation dates & check for files
    nodd_file_list = []
    nodd_total_size = 0
    for date in date_generated:
        file_date = date.strftime("%Y%m%d")
        year = file_date[:4]
        prod_path = "noaa-cdr-aerosol-optical-thickness-pds/data/daily/" + year + "/"
        patt = "AOT_AVHRR_*_daily-avg_"
        file_names = fs.glob(prod_path + patt + file_date + "_*.nc")
        # If file exists, add path to list and add file size to total
        if file_names:
            nodd_file_list.extend(file_names)
            nodd_total_size = nodd_total_size + sum(fs.size(f) for f in file_names)
        else:
            msg = "File does not exist on AWS: " + prod_path + patt + file_date + "_*.nc"
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
    import warnings

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
            nodd_total_size = nodd_total_size + sum(fs.size(f) for f in file_names)
        else:
            msg = "File does not exist on AWS: " + prod_path + patt + file_date + "_*.nc"
            if warning:
                warnings.warn(msg)
                nodd_file_list.append(None)
            else:
                raise ValueError(msg)

    return nodd_file_list, nodd_total_size


def open_dataset(date, averaging_time="daily"):
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
    import pandas as pd
    import s3fs
    import xarray as xr

    if isinstance(date, str):
        date_generated = [pd.Timestamp(date)]
    else:
        date_generated = [date]

    # Access AWS using anonymous credentials
    fs = s3fs.S3FileSystem(anon=True)

    if averaging_time.lower() == "monthly":
        file_list, _ = create_monthly_aod_list(date_generated, fs)
    elif averaging_time.lower() == "daily":
        file_list, _ = create_daily_aod_list(date_generated, fs)
    else:
        raise ValueError(
            f"Invalid input for 'averaging_time' {averaging_time!r}: "
            "Valid values are 'daily' or 'monthly'"
        )

    if len(file_list) == 0 or all(f is None for f in file_list):
        raise ValueError(f"Files not available for product and date: {date_generated[0]}")

    aws_file = fs.open(file_list[0])

    dset = xr.open_dataset(aws_file)

    return dset


def open_mfdataset(dates, averaging_time="daily", error_missing=False):
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

    import pandas as pd
    import s3fs
    import xarray as xr

    if isinstance(dates, Iterable) and not isinstance(dates, str):
        dates = pd.DatetimeIndex(dates)
    else:
        dates = pd.DatetimeIndex([dates])

    # Access AWS using anonymous credentials
    fs = s3fs.S3FileSystem(anon=True)

    if averaging_time.lower() == "monthly":
        file_list, _ = create_monthly_aod_list(dates, fs, warning=not error_missing)
    elif averaging_time.lower() == "daily":
        file_list, _ = create_daily_aod_list(dates, fs, warning=not error_missing)
    else:
        raise ValueError(
            f"Invalid input for 'averaging_time' {averaging_time!r}: "
            "Valid values are 'daily' or 'monthly'"
        )

    if len(file_list) == 0 or all(f is None for f in file_list):
        raise ValueError(f"Files not available for product and dates: {dates}")

    aws_files = [fs.open(f) for f in file_list if f is not None]

    dset = xr.open_mfdataset(aws_files, concat_dim="time", combine="nested")

    return dset
