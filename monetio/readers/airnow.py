from datetime import datetime
from typing import Any, List, Optional, Union

import pandas as pd
import xarray as xr

from .core import PointReader
from .drivers import PandasDriver


class AirNowReader(PointReader):
    """Reader for AirNow hourly and daily data."""

    def build_urls(self, dates: pd.DatetimeIndex, daily: bool = False) -> List[str]:
        """Construct AirNow file URLs for `dates`.

        Parameters
        ----------
        dates : pd.DatetimeIndex
            Dates to retrieve.
        daily : bool, optional
            Whether to retrieve daily data instead of hourly.

        Returns
        -------
        List[str]
            List of URLs.
        """
        if daily:
            dates = dates.floor("D").unique()
        else:
            dates = dates.floor("H").unique()

        urls = []
        base_url = "https://s3-us-west-1.amazonaws.com//files.airnowtech.org/airnow/"
        for dt in dates:
            if daily:
                fname = "daily_data.dat"
            else:
                fname = dt.strftime(r"HourlyData_%Y%m%d%H.dat")
            url = base_url + dt.strftime(r"%Y/%Y%m%d/") + fname
            urls.append(url)
        return urls

    def read_airnow(self, f: str, daily: bool = False) -> pd.DataFrame:
        """Read a single AirNow file.

        Parameters
        ----------
        f : str
            File path or URL.
        daily : bool, optional
            Whether the file contains daily data.

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
        daily_cols = ["date", "siteid", "site", "variable", "units", "obs", "hours", "source"]

        try:
            # We use a try-except to handle potentially empty or malformed files
            df = pd.read_csv(
                f,
                delimiter="|",
                header=None,
                encoding="ISO-8859-1",
                on_bad_lines="warn",
            )
        except Exception:
            return pd.DataFrame(columns=daily_cols if daily else hourly_cols)

        if df.empty:
            return pd.DataFrame(columns=daily_cols if daily else hourly_cols)

        ncols = df.columns.size
        if ncols == len(hourly_cols):
            df.columns = hourly_cols
        elif ncols == len(daily_cols):
            df.columns = daily_cols
        else:
            # Fallback or error?
            return pd.DataFrame(columns=daily_cols if daily else hourly_cols)

        df["obs"] = pd.to_numeric(df["obs"], errors="coerce")
        df["siteid"] = df["siteid"].astype(str).str.zfill(9)

        return df

    def _post_process(
        self,
        df: Union[pd.DataFrame, "Any"],
        daily: bool = False,
        bad_utcoffset: str = "drop",
        **kwargs,
    ) -> Union[pd.DataFrame, "Any"]:
        """Post-process the combined AirNow DataFrame.

        Parameters
        ----------
        df : Union[pd.DataFrame, dask.dataframe.DataFrame]
            The combined DataFrame.
        daily : bool, optional
            Whether the data is daily.
        bad_utcoffset : str, optional
            How to handle bad UTC offsets.
        **kwargs
            Additional arguments.

        Returns
        -------
        Union[pd.DataFrame, dask.dataframe.DataFrame]
            The post-processed DataFrame.
        """
        import numpy as np

        # Vectorized time parsing
        if daily:
            if hasattr(df, "compute"):  # Dask
                import dask.dataframe as dd

                df["time"] = dd.to_datetime(df["date"], format=r"%m/%d/%y")
            else:
                df["time"] = pd.to_datetime(df["date"], format=r"%m/%d/%y")
        else:
            if hasattr(df, "compute"):  # Dask
                import dask.dataframe as dd

                df["time"] = dd.to_datetime(df["date"] + " " + df["time"], format=r"%m/%d/%y %H:%M")
            else:
                df["time"] = pd.to_datetime(df["date"] + " " + df["time"], format=r"%m/%d/%y %H:%M")

            # Calculate time_local
            # Using astype('timedelta64[h]') is often more robust for Dask
            df["time_local"] = df["time"] + df["utcoffset"].astype("timedelta64[h]")

        df = df.drop(columns=["date"], errors="ignore")
        if not daily:
            df = df.drop(columns=["time_original"], errors="ignore")  # if it existed

        # Add station locations
        from ..obs.epa_util import read_monitor_file

        try:
            monitor_df = read_monitor_file(airnow=True).drop_duplicates(subset=["siteid"])
        except Exception:
            # Fallback if network or file is missing
            monitor_df = pd.DataFrame(columns=["siteid", "latitude", "longitude"])
        # We only need a few columns from monitor_df to avoid bloat
        # and we need to make sure we don't have name collisions
        df = df.merge(monitor_df, on="siteid", how="left")

        # Filter bad values
        df["obs"] = df["obs"].where((df["obs"] <= 3000) & (df["obs"] >= 0), np.nan)

        # Handle bad UTC offsets
        if "utcoffset" in df.columns and "longitude" in df.columns:
            # Identify bad rows: utcoffset == 0 but longitude is far from Prime Meridian
            # For Dask, we use .where or filtering
            if bad_utcoffset == "drop":
                df = df.loc[~((df["utcoffset"] == 0) & (df["longitude"].abs() > 20))]
            elif bad_utcoffset == "null":
                # Use where to set to NaN
                df["utcoffset"] = df["utcoffset"].where(
                    ~((df["utcoffset"] == 0) & (df["longitude"].abs() > 20)), np.nan
                )
            # 'fix' is complex for Dask without timezonefinder being vectorized/delayed
            # skipping 'fix' for now in post_process if it's too heavy

        # Update history
        df = self.update_history(df, "Loaded AirNow data and added station metadata.")

        return df

    def open_dataset(
        self,
        files: Optional[Union[str, List[str]]] = None,
        dates: Optional[Union[pd.DatetimeIndex, List[datetime]]] = None,
        daily: bool = False,
        lazy: bool = False,
        **kwargs,
    ) -> xr.Dataset:
        """Retrieve and load AirNow data.

        Parameters
        ----------
        files : Optional[Union[str, List[str]]], optional
            File paths or URLs.
        dates : Optional[Union[pd.DatetimeIndex, List[datetime]]], optional
            Dates to retrieve.
        daily : bool, optional
            Whether to retrieve daily data.
        lazy : bool, optional
            Whether to load lazily with Dask.
        **kwargs
            Additional arguments passed to PandasDriver.open_dataset.

        Returns
        -------
        xr.Dataset
            The loaded dataset.
        """
        if files is None:
            if dates is None:
                raise ValueError("Either files or dates must be provided.")
            if not isinstance(dates, pd.DatetimeIndex):
                dates = pd.DatetimeIndex(dates)
            files = self.build_urls(dates, daily=daily)

        if isinstance(files, str):
            files = [files]

        df = PandasDriver.open_dataset(
            files,
            read_function=self.read_airnow,
            post_process_function=self._post_process,
            lazy=lazy,
            daily=daily,
            **kwargs,
        )

        return self.to_xarray(df)


def open_dataset(
    files: Optional[Union[str, List[str]]] = None,
    dates: Optional[Union[pd.DatetimeIndex, List[datetime]]] = None,
    daily: bool = False,
    lazy: bool = False,
    **kwargs,
) -> xr.Dataset:
    """Retrieve and load AirNow data.

    Parameters
    ----------
    files : Optional[Union[str, List[str]]], optional
        File paths or URLs.
    dates : Optional[Union[pd.DatetimeIndex, List[datetime]]], optional
        Dates to retrieve.
    daily : bool, optional
        Whether to retrieve daily data.
    lazy : bool, optional
        Whether to load lazily with Dask.
    **kwargs
        Additional arguments.

    Returns
    -------
    xr.Dataset
    """
    return AirNowReader().open_dataset(files=files, dates=dates, daily=daily, lazy=lazy, **kwargs)
