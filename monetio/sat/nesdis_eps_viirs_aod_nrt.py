import pandas as pd


def build_urls(dates, *, daily=True, data_resolution=0.1, satellite="NOAA20"):
    """Construct URLs for downloading NEPS data.

    Parameters
    ----------
    dates : pd.DatetimeIndex or iterable of datetime
        Dates to download data for.
    daily : bool, optional
        Whether to download daily (default) or monthly data.
    data_resolution : float, optional
        Resolution of data in degrees (0.1 or 0.25).
    satellite : str, optional
        Satellite platform, 'SNPP' or 'NOAA20'.

    Returns
    -------
    pd.Series
        Series with URLs and corresponding file names.

    Notes
    -----
    The `res` and `sat` parameters are only used for sub-daily data.
    """
    import warnings
    from collections.abc import Iterable

    if isinstance(dates, Iterable) and not isinstance(dates, str):
        dates = pd.DatetimeIndex(dates)
    else:
        dates = pd.DatetimeIndex([dates])

    if daily:
        dates = dates.floor("D").unique()
    else:  # monthly
        dates = dates.to_period("M").to_timestamp().unique()

    if data_resolution != 0.25 and not daily:
        warnings.warn(
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

    # Note: files needed for comparison
    urls = pd.Series(urls, index=None)
    fnames = pd.Series(fnames, index=None)

    return urls, fnames


def open_dataset(date, *, satellite="NOAA20", data_resolution=0.1, daily=True):
    """
    Parameters
    ----------
    date : str or datetime-like
        The date for which to open the dataset.
    """
    from io import BytesIO

    import pandas as pd
    import requests
    import xarray as xr

    if not isinstance(date, pd.Timestamp):
        d = pd.to_datetime(date)
    else:
        d = date

    if satellite.lower() not in ("noaa20", "snpp"):
        raise ValueError(
            f"Invalid input for 'satellite' {satellite!r}: " "Valid values are 'NOAA20' or 'SNPP'"
        )

    if data_resolution not in {0.1, 0.25}:
        raise ValueError(
            f"Invalid input for 'data_resolution' {data_resolution!r}: "
            "Valid values are 0.1 or 0.25"
        )

    urls, _ = build_urls(d, satellite=satellite, data_resolution=data_resolution, daily=daily)

    r = requests.get(urls[0], stream=True)
    r.raise_for_status()
    dset = xr.open_dataset(BytesIO(r.content))

    dset = dset.expand_dims(time=[d]).set_coords(["time"])

    return dset


def open_mfdataset(dates, satellite="NOAA20", data_resolution=0.1, daily=True, error_missing=False):
    import warnings
    from collections.abc import Iterable
    from io import BytesIO

    import pandas as pd
    import requests
    import xarray as xr

    if isinstance(dates, Iterable) and not isinstance(dates, str):
        dates = pd.DatetimeIndex(dates)
    else:
        dates = pd.DatetimeIndex([dates])

    if satellite.lower() not in ("noaa20", "snpp"):
        raise ValueError(
            f"Invalid input for 'satellite' {satellite!r}: " "Valid values are 'NOAA20' or 'SNPP'"
        )

    if data_resolution not in {0.1, 0.25}:
        raise ValueError(
            f"Invalid input for 'data_resolution' {data_resolution!r}: "
            "Valid values are 0.1 or 0.25"
        )

    urls, _ = build_urls(dates, satellite=satellite, data_resolution=data_resolution, daily=daily)

    dsets = []
    for url, date in zip(urls, dates):
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            msg = f"Failed to access file on NESDIS FTP server: {url}"
            if error_missing:
                raise RuntimeError(msg)
            else:
                warnings.warn(msg)
        else:
            ds = xr.open_dataset(BytesIO(r.content)).expand_dims(time=[date]).set_coords(["time"])
            dsets.append(ds)

    if len(dsets) == 0:
        raise ValueError(f"Files not available for product and dates: {dates}")

    dset = xr.concat(dsets, dim="time")

    return dset
