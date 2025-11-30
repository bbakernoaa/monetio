"""AirNow Reader"""

import os
from datetime import datetime

import pandas as pd
import dask
import dask.dataframe as dd
from numpy import nan

from monetio.readers.base import PointReader, register_reader
from monetio.obs.epa_util import read_monitor_file
from monetio.util import long_to_wide


# Global variable to hold TimezoneFinder instance
_TFinder = None

@register_reader("airnow")
class AirNowReader(PointReader):
    def open_dataset(self,
                     files=None, # Note: 'files' here might actually be dates for AirNow
                     dates=None, # Explicit dates argument since AirNow works off dates
                     download=False,
                     wide_fmt=True,
                     n_procs=1,
                     daily=False,
                     bad_utcoffset="drop",
                     **kwargs):
        """
        Retrieve and load AirNow data as a DataFrame.

        If `files` is provided, it's treated as a list of file paths (local or S3).
        If `dates` is provided (legacy behavior), it constructs the URLs.

        Args:
            files: List of specific file paths (optional).
            dates: List of datetimes or similar (optional).
            download: Whether to download files locally (if constructing URLs).
            wide_fmt: Return in wide format.
            n_procs: Dask workers.
            daily: Daily data vs Hourly.
            bad_utcoffset: How to handle bad UTC offsets.

        Returns:
            pd.DataFrame
        """

        # Handle the 'dates' logic which is unique to AirNow's API style
        # In the new architecture, we generally expect 'files' to be passed.
        # But for backward compatibility/ease of use with online sources, we keep the date logic.

        if files is None and dates is not None:
            # Construct URLs from dates
            urls, fnames = build_urls(dates, daily=daily)

            if download:
                # We need to download them first
                # This logic was in aggregate_files
                for url, fname in zip(urls, fnames):
                    retrieve(url, fname)
                files = fnames.tolist()
            else:
                files = urls.tolist()

        if not files:
             raise ValueError("Must provide either 'files' or 'dates'.")

        # Use PandasDriver via self.driver
        # However, the original logic used Dask to read multiple CSVs in parallel.
        # PandasDriver.open does a loop and concat.
        # For AirNow, which can have many small files, the Dask approach is better.
        # But for the sake of the "Driver" abstraction, we should arguably use the driver.
        # But the driver is simple.

        # If we want to stick to the Driver abstraction strictly:
        # df = self.driver.open(files, read_method='read_csv', ... options ...)
        # But we need custom parsing options.

        # Let's implement the aggregation logic here, using the driver for individual file reads if possible,
        # OR just re-implement the dask logic if it's critical for performance.
        # The prompt said "Port logic...". The logic uses dask.delayed(read_csv).
        # Our PandasDriver doesn't support dask.delayed yet.

        # Let's keep the Dask logic for performance, but maybe wrapping it cleaner.
        # Or, we can just accept that PointReader.driver is for simple cases and override for complex ones.

        # Re-implementing aggregate_files logic inside open_dataset
        print("Aggregating AIRNOW files...")

        # We define a custom read function that matches the old read_csv
        def _read_helper(fn):
            return read_airnow_csv(fn, daily=daily)

        # Use dask for parallel reading as in original
        dfs = [dask.delayed(_read_helper)(f) for f in files]
        dff = dd.from_delayed(dfs)
        df = dff.compute(num_workers=n_procs).reset_index()

        # Datetime conversion
        if daily:
            df["time"] = pd.to_datetime(df.date, format=r"%m/%d/%y", exact=True)
        else:
             # TODO: move to read_csv? (and some of this other stuff too?)
            df["time"] = pd.to_datetime(
                df.date + " " + df.time, format=r"%m/%d/%y %H:%M", exact=True
            )
            df["time_local"] = df.time + pd.to_timedelta(df.utcoffset, unit="H")

        df.drop(["date"], axis=1, inplace=True)

        print("    Adding in Meta-data")
        df = get_station_locations(df)

        savecols = [
            "time", "siteid", "site", "utcoffset", "variable", "units", "obs",
            "time_local", "latitude", "longitude", "cmsa_name", "msa_code",
            "msa_name", "state_name", "epa_region",
        ]

        if daily:
            cols = [col for col in savecols if col not in {"time_local", "utcoffset"}]
            df = df[cols]
        else:
            df = df[savecols]

        df.drop_duplicates(inplace=True)
        df = filter_bad_values(df, bad_utcoffset=bad_utcoffset)
        df = df.reset_index(drop=True)

        # Post-processing (Wide format)
        if wide_fmt:
             df = (
                long_to_wide(df)
                .drop_duplicates(subset=["time", "latitude", "longitude", "siteid"])
                .reset_index(drop=True)
            )

        return self.harmonize(df)


# -----------------------------------------------------------------------------
# Helper functions ported from monetio/obs/airnow.py
# -----------------------------------------------------------------------------

def build_urls(dates, *, daily=False):
    dates = pd.DatetimeIndex(dates)
    if daily:
        dates = dates.floor("D").unique()
    else:  # hourly
        dates = dates.floor("H").unique()

    urls = []
    fnames = []
    print("Building AIRNOW URLs...")
    base_url = "https://s3-us-west-1.amazonaws.com//files.airnowtech.org/airnow/"
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
    import requests

    if not os.path.isfile(fname):
        print("\n Retrieving: " + fname)
        print(url)
        print("\n")
        r = requests.get(url)
        r.raise_for_status()
        with open(fname, "wb") as f:
            f.write(r.content)
    else:
        print("\n File Exists: " + fname)


def read_airnow_csv(fn, daily=False):
    hourly_cols = [
        "date", "time", "siteid", "site", "utcoffset", "variable", "units", "obs", "source",
    ]
    daily_cols = ["date", "siteid", "site", "variable", "units", "obs", "hours", "source"]

    try:
        # Check if it's an S3 URL or local file
        # If it's S3, we might need to open it.
        # But pandas read_csv can handle HTTP URLs directly if they are public.
        # The URLs generated are https URLs to S3.

        dft = pd.read_csv(
            fn,
            delimiter="|",
            header=None,
            encoding="ISO-8859-1",
            on_bad_lines="warn",
        )
    except Exception:
        dft = pd.DataFrame(columns=hourly_cols)

    # Assign column names
    ncols = dft.columns.size
    if ncols == len(hourly_cols):
        dft.columns = hourly_cols
    elif ncols == len(hourly_cols) - 1:  # daily data
        daily = True # Set local daily flag if inferred
        dft.columns = daily_cols
    else:
        # Fallback or error
        # For now raise as in original
        raise Exception(f"unexpected number of columns: {ncols}")

    dft["obs"] = dft.obs.astype(float)
    dft["siteid"] = dft.siteid.str.zfill(9)

    if not daily and "utcoffset" in dft.columns:
        dft["utcoffset"] = dft.utcoffset.astype(int)

    return dft


def filter_bad_values(df, *, max=3000, bad_utcoffset="drop"):
    from numpy import nan

    df.loc[(df.obs > max) | (df.obs < 0), "obs"] = nan

    # Bad UTC offsets (GH #86)
    if "utcoffset" in df.columns:
        bad_rows = df.query("utcoffset == 0 and abs(longitude) > 20")
        if bad_utcoffset == "null":
            df.loc[bad_rows.index, "utcoffset"] = nan
        elif bad_utcoffset == "drop":
            df.drop(bad_rows.index, inplace=True)
        elif bad_utcoffset == "fix":
            df.loc[bad_rows.index, "utcoffset"] = bad_rows.apply(
                lambda row: get_utcoffset(row.latitude, row.longitude),
                axis="columns",
            )
        elif bad_utcoffset == "leave":
            pass
        else:
            raise ValueError("`bad_utcoffset` must be one of: 'null', 'drop', 'fix', 'leave'")

    return df


def get_utcoffset(lat, lon):
    import warnings

    try:
        import pytz
        import timezonefinder
    except ImportError:
        warnings.warn(
            "timezonefinder and/or pytz not installed, guessing UTC offset based on longitude"
        )
        do_guess = True
    else:
        do_guess = False

    if do_guess:
        lon_ = (lon + 180) % 360 - 180
        return round(lon_ / 15, 0)

    else:
        global _TFinder

        if _TFinder is None:
            _TFinder = timezonefinder.TimezoneFinder(in_memory=True)

        finder = _TFinder
        tz_str = finder.timezone_at(lng=lon, lat=lat)
        tz = pytz.timezone(tz_str)
        uo = tz.utcoffset(datetime(2020, 1, 1), is_dst=False).total_seconds() / 3600
        return uo


def get_station_locations(df):
    # Helper to merge station metadata
    monitor_df = read_monitor_file(airnow=True)
    df = df.merge(monitor_df.drop_duplicates(), on="siteid", how="left", copy=False)
    return df
