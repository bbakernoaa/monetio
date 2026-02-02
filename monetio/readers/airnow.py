"""AirNow Reader"""

import os
from datetime import datetime
from functools import lru_cache

import dask
import dask.dataframe as dd
import pandas as pd
from numpy import nan

from monetio.obs.epa_util import read_monitor_file
from monetio.readers.base import PointReader, register_reader
from monetio.util import long_to_wide

from .drivers import FileUtility


@register_reader("airnow")
class AirNowReader(PointReader):
    def open_dataset(
        self,
        files=None,
        dates=None,
        download=False,
        wide_fmt=True,
        n_procs=1,
        daily=False,
        bad_utcoffset="drop",
        as_xarray=False,
        **kwargs,
    ):
        """
        Retrieve and load AirNow data as a DataFrame or xarray Dataset.
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

        print("Aggregating AIRNOW files...")

        # We define a custom read function that matches the old read_csv
        # Pass storage options if S3
        storage_options = kwargs.get("storage_options", {})
        if not storage_options and any(f.startswith("s3://") for f in files):
            storage_options = {"anon": True}

        def _read_helper(fn):
            return read_airnow_csv(fn, daily=daily, storage_options=storage_options)

        dfs = [dask.delayed(_read_helper)(f) for f in files]
        dff = dd.from_delayed(dfs)
        df = dff.compute(num_workers=n_procs).reset_index()

        if daily:
            df["time"] = pd.to_datetime(df.date, format=r"%m/%d/%y", exact=True)
        else:
            df["time"] = pd.to_datetime(
                df.date + " " + df.time, format=r"%m/%d/%y %H:%M", exact=True
            )
            df["time_local"] = df.time + pd.to_timedelta(df.utcoffset, unit="h")

        df.drop(["date"], axis=1, inplace=True)

        print("    Adding in Meta-data")
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
            df = df[cols]
        else:
            df = df[savecols]

        df.drop_duplicates(inplace=True)
        df = filter_bad_values(df, bad_utcoffset=bad_utcoffset)
        df = df.reset_index(drop=True)

        if wide_fmt:
            df = (
                long_to_wide(df)
                .drop_duplicates(subset=["time", "latitude", "longitude", "siteid"])
                .reset_index(drop=True)
            )

        df = self.harmonize(df)
        if as_xarray:
            return self.to_xarray(df)

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


def read_airnow_csv(fn, daily=False, storage_options=None):
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
    from numpy import nan

    df.loc[(df.obs > max) | (df.obs < 0), "obs"] = nan

    if "utcoffset" in df.columns:
        bad_rows = df.query("utcoffset == 0 and abs(longitude) > 20")
        if bad_utcoffset == "null":
            df.loc[bad_rows.index, "utcoffset"] = nan
        elif bad_utcoffset == "drop":
            df.drop(bad_rows.index, inplace=True)
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
    monitor_df = read_monitor_file(airnow=True)
    df = df.merge(monitor_df.drop_duplicates(), on="siteid", how="left", copy=False)
    return df
