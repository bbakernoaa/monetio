"""AirNow Reader"""

import os
from datetime import datetime
from functools import lru_cache, partial
from typing import TYPE_CHECKING, List, Union

import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    import dask.dataframe as dd
from numpy import nan

from monetio.obs.epa_util import read_monitor_file
from monetio.readers.base import PointReader, register_reader
from monetio.util import long_to_wide

from .drivers import FileUtility


@register_reader("airnow")
class AirNowReader(PointReader):
    def open_dataset(
        self,
        files: Union[str, List[str]] = None,
        dates: Union[pd.DatetimeIndex, List[datetime], datetime, str] = None,
        download: bool = False,
        wide_fmt: bool = True,
        n_procs: int = 1,
        daily: bool = False,
        bad_utcoffset: str = "drop",
        as_xarray: bool = True,
        lazy: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, xr.Dataset, "dd.DataFrame"]:
        """
        Retrieve and load AirNow data.

        Parameters
        ----------
        files : Union[str, List[str]], optional
            File path, list of paths, or glob pattern.
        dates : Union[pd.DatetimeIndex, List[datetime], datetime, str], optional
            Dates to retrieve if files are not provided.
        download : bool, optional
            Whether to download files to local directory, by default False.
        wide_fmt : bool, optional
            Whether to return data in wide format, by default True.
        n_procs : int, optional
            Number of processors for dask compute (if not lazy), by default 1.
        daily : bool, optional
            Whether to load daily data instead of hourly, by default False.
        bad_utcoffset : str, optional
            How to handle sites with zero UTC offset and large longitude.
            Options: 'drop', 'null', 'fix', 'leave'. By default 'drop'.
        as_xarray : bool, optional
            Whether to return an xarray.Dataset, by default True.
        lazy : bool, optional
            Whether to return a dask-backed object, by default False.
        **kwargs : dict
            Additional arguments passed to the driver.

        Returns
        -------
        Union[pd.DataFrame, xr.Dataset, dd.DataFrame]
            The loaded AirNow data.
        """

        if files is None and dates is not None:
            # Construct URLs from dates
            urls, fnames = build_urls(dates, daily=daily)

            if download:
                for url, fname in zip(urls, fnames):
                    retrieve(url, fname)
                files = fnames.tolist()
            else:
                files = urls.tolist()

        if not files:
            raise ValueError("Must provide either 'files' or 'dates'.")

        # Define per-file preprocessing
        storage_options = kwargs.get("storage_options", {})
        if not storage_options and any(str(f).startswith("s3://") for f in files):
            storage_options = {"anon": True}

        read_func = partial(read_airnow_csv, daily=daily, storage_options=storage_options)

        # Use base class to open
        df = super().open_dataset(
            files,
            read_method=read_func,
            as_xarray=False,  # We do conversion manually after post-processing
            lazy=lazy,
            **kwargs,
        )

        # Post-processing (Backend-agnostic)
        df = self._post_process(df, daily=daily, wide_fmt=wide_fmt, bad_utcoffset=bad_utcoffset)
        df = self.harmonize(df)

        if not lazy and hasattr(df, "compute"):
            df = df.compute(num_workers=n_procs)

        if as_xarray:
            ds = self.to_xarray(df)
            # Update history
            history = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read AirNow data."
            if "history" in ds.attrs:
                ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
            else:
                ds.attrs["history"] = history
            return ds

        return df

    def _post_process(self, df, daily=False, wide_fmt=True, bad_utcoffset="drop"):
        """Internal post-processing logic."""
        import pandas as pd

        # Check if dask
        try:
            import dask.dataframe as dd

            is_dask = isinstance(df, dd.DataFrame)
        except ImportError:
            is_dask = False

        if is_dask:
            # Dask operations
            if daily:
                df["time"] = dd.to_datetime(df.date, format=r"%m/%d/%y")
            else:
                df["time"] = dd.to_datetime(df.date + " " + df.time, format=r"%m/%d/%y %H:%M")
                df["time_local"] = df.time + dd.to_timedelta(df.utcoffset, unit="h")
        else:
            # Pandas operations
            if daily:
                df["time"] = pd.to_datetime(df.date, format=r"%m/%d/%y", exact=True)
            else:
                df["time"] = pd.to_datetime(
                    df.date + " " + df.time, format=r"%m/%d/%y %H:%M", exact=True
                )
                df["time_local"] = df.time + pd.to_timedelta(df.utcoffset, unit="h")

        df = df.drop(columns=["date"])

        # Metadata
        df = get_station_locations(df)

        savecols = [
            "time",
            "siteid",
            "site",
            "utcoffset",
            "variable",
            "units",
            "obs",
            "time_local",
            "latitude",
            "longitude",
            "cmsa_name",
            "msa_code",
            "msa_name",
            "state_name",
            "epa_region",
        ]

        if daily:
            cols = [col for col in savecols if col not in {"time_local", "utcoffset"}]
            df = df[[c for c in cols if c in df.columns]]
        else:
            df = df[[c for c in savecols if c in df.columns]]

        df = df.drop_duplicates()
        df = filter_bad_values(df, bad_utcoffset=bad_utcoffset)

        if wide_fmt:
            # Note: long_to_wide uses pivot_table which might trigger compute on Dask
            df = long_to_wide(df)
            # drop_duplicates after wide might also be expensive
            subset = [c for c in ["time", "latitude", "longitude", "siteid"] if c in df.columns]
            df = df.drop_duplicates(subset=subset)

        return df


# -----------------------------------------------------------------------------
# Helper functions ported from monetio/obs/airnow.py
# -----------------------------------------------------------------------------


def build_urls(dates, *, daily=False):
    dates = pd.DatetimeIndex(dates)
    if daily:
        dates = dates.floor("D").unique()
    else:  # hourly
        dates = dates.floor("h").unique()

    urls = []
    fnames = []
    print("Building AIRNOW URLs...")
    # Use S3 bucket directly
    base_url = "s3://files.airnowtech.org/airnow/"
    for dt in dates:
        if daily:
            fname = "daily_data.dat"
        else:
            fname = dt.strftime(r"HourlyData_%Y%m%d%H.dat")
        url = base_url + dt.strftime(r"%Y/%Y%m%d/") + fname
        urls.append(url)
        fnames.append(fname)

    return pd.Series(urls, index=None), pd.Series(fnames, index=None)


def retrieve(url, fname):
    if not os.path.isfile(fname):
        print("\n Retrieving: " + fname)
        print(url)

        if url.startswith("s3://"):
            fs = FileUtility.get_fs(url)
            fs.get(url, fname)
        elif url.startswith("http"):
            import requests

            r = requests.get(url)
            r.raise_for_status()
            with open(fname, "wb") as f:
                f.write(r.content)
        else:
            # Local file copy?
            pass

        print("\n Retrieved")
    else:
        print("\n File Exists: " + fname)


def read_airnow_csv(fn, daily=False, storage_options=None, **kwargs):
    """
    Read a single AirNow CSV file.

    Parameters
    ----------
    fn : str
        File path or URL.
    daily : bool, optional
        Whether the file contains daily data, by default False.
    storage_options : dict, optional
        Storage options for fsspec, by default None.
    **kwargs : dict
        Additional arguments passed to pd.read_csv.

    Returns
    -------
    pd.DataFrame
        The loaded data.
    """
    hourly_cols = [
        "date",
        "time",
        "siteid",
        "site",
        "utcoffset",
        "variable",
        "units",
        "obs",
        "source",
    ]
    daily_cols = [
        "date",
        "siteid",
        "site",
        "variable",
        "units",
        "obs",
        "hours",
        "source",
    ]

    try:
        dft = pd.read_csv(
            fn,
            delimiter="|",
            header=None,
            encoding="ISO-8859-1",
            on_bad_lines="warn",
            storage_options=storage_options,
        )
    except Exception:
        dft = pd.DataFrame(columns=hourly_cols)

    ncols = dft.columns.size
    if ncols == len(hourly_cols):
        dft.columns = hourly_cols
    elif ncols == len(hourly_cols) - 1:
        daily = True
        dft.columns = daily_cols
    else:
        # Return empty with correct cols if mismatch
        # Or raise
        if daily:
            return pd.DataFrame(columns=daily_cols)
        else:
            return pd.DataFrame(columns=hourly_cols)

    dft["obs"] = dft.obs.astype(float)
    dft["siteid"] = dft.siteid.str.zfill(9)

    if not daily and "utcoffset" in dft.columns:
        dft["utcoffset"] = dft.utcoffset.astype(int)

    return dft


def filter_bad_values(df, *, max=3000, bad_utcoffset="drop"):
    """
    Filter bad values and handle zero UTC offsets.

    Parameters
    ----------
    df : Union[pd.DataFrame, dd.DataFrame]
        Input dataframe.
    max : int, optional
        Maximum allowed observation value, by default 3000.
    bad_utcoffset : str, optional
        How to handle sites with zero UTC offset and large longitude,
        by default "drop".

    Returns
    -------
    Union[pd.DataFrame, dd.DataFrame]
        Filtered dataframe.
    """
    from numpy import nan

    df["obs"] = df["obs"].where((df.obs <= max) & (df.obs >= 0), nan)

    if "utcoffset" in df.columns:
        bad_rows = df.query("utcoffset == 0 and abs(longitude) > 20")
        if bad_utcoffset == "null":
            # For dask compatibility
            df["utcoffset"] = df["utcoffset"].where(
                ~((df.utcoffset == 0) & (df.longitude.abs() > 20)), nan
            )
        elif bad_utcoffset == "drop":
            df = df.loc[~((df.utcoffset == 0) & (df.longitude.abs() > 20))]
        elif bad_utcoffset == "fix":
            # TimezoneFinder is slow, so only call it for unique locations
            unique_locs = bad_rows.drop_duplicates(subset=["latitude", "longitude"])
            tz_map = {
                (lat, lon): get_utcoffset(lat, lon)
                for lat, lon in zip(unique_locs.latitude, unique_locs.longitude)
            }
            s_offset = bad_rows.apply(
                lambda row: tz_map.get((row.latitude, row.longitude)),
                axis="columns",
            )
            df.loc[bad_rows.index, "utcoffset"] = s_offset
        elif bad_utcoffset == "leave":
            pass
        else:
            raise ValueError("`bad_utcoffset` must be one of: 'null', 'drop', 'fix', 'leave'")

    return df


@lru_cache(maxsize=1)
def _get_tf(*, in_memory=True):
    import timezonefinder

    return timezonefinder.TimezoneFinder(in_memory=in_memory)


@lru_cache(maxsize=1024)
def get_utcoffset(lat, lon):
    import warnings

    try:
        import pytz
    except ImportError:
        warnings.warn("pytz not installed, guessing UTC offset based on longitude")
        do_guess = True
    else:
        do_guess = False

    if do_guess:
        lon_ = (lon + 180) % 360 - 180
        return round(lon_ / 15, 0)

    else:
        finder = _get_tf()
        tz_str = finder.timezone_at(lng=lon, lat=lat)
        if tz_str:
            tz = pytz.timezone(tz_str)
            uo = tz.utcoffset(datetime(2020, 1, 1), is_dst=False).total_seconds() / 3600
            return uo
        else:
            return nan


def get_station_locations(df):
    """
    Add site metadata to the dataframe.

    Parameters
    ----------
    df : Union[pd.DataFrame, dd.DataFrame]
        Input dataframe.

    Returns
    -------
    Union[pd.DataFrame, dd.DataFrame]
        Dataframe with site metadata.
    """
    monitor_df = read_monitor_file(airnow=True)
    # Check if dask
    try:
        import dask.dataframe as dd

        is_dask = isinstance(df, dd.DataFrame)
    except ImportError:
        is_dask = False

    if is_dask:
        df = df.merge(monitor_df.drop_duplicates(), on="siteid", how="left")
    else:
        df = df.merge(monitor_df.drop_duplicates(), on="siteid", how="left", copy=False)
    return df
