import pandas as pd

from .core import PointReader
from .drivers import PandasDriver


def read_airnow(fn):
    """Read an AirNow file.

    Parameters
    ----------
    fn : str
        File name or URL.

    Returns
    -------
    pandas.DataFrame
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
        # Note: using on_bad_lines instead of error_bad_lines/warn_bad_lines for modern pandas
        df = pd.read_csv(
            fn,
            delimiter="|",
            header=None,
            on_bad_lines="warn",
            encoding="ISO-8859-1",
        )
    except FileNotFoundError:
        # Expected error if file is missing
        return pd.DataFrame(columns=hourly_cols)
    except Exception as e:
        # Log unexpected errors
        import logging

        logging.warning(f"Error reading {fn}: {e}")
        return pd.DataFrame(columns=hourly_cols)

    ncols = df.columns.size
    if ncols == len(hourly_cols):
        df.columns = hourly_cols
    elif ncols == len(hourly_cols) - 1:
        df.columns = daily_cols
    else:
        # Fallback for unexpected columns
        return pd.DataFrame(columns=hourly_cols)

    df["obs"] = pd.to_numeric(df["obs"], errors="coerce")
    df["siteid"] = df["siteid"].astype(str).str.zfill(9)

    return df


class AirNowReader(PointReader):
    """AirNow reader following the Aero Protocol."""

    def __init__(self):
        super().__init__()
        self.driver = PandasDriver()

    def open_dataset(self, dates=None, files=None, *, lazy=True, daily=False, **kwargs):
        """Open AirNow dataset.

        Parameters
        ----------
        dates : array-like, optional
            Dates to retrieve.
        files : list of str, optional
            List of files to open. If provided, dates are ignored for file discovery.
        lazy : bool, optional
            Whether to load lazily.
        daily : bool, optional
            Whether to load daily data.
        **kwargs : dict
            Additional arguments.

        Returns
        -------
        xarray.Dataset
        """
        if files is None and dates is not None:
            from ..obs.airnow import build_urls

            urls, _ = build_urls(dates, daily=daily)
            files = urls.tolist()

        if files is None:
            raise ValueError("Either 'dates' or 'files' must be provided.")

        # 1. Load data via driver
        df = self.driver.open_dataset(files, reader=read_airnow, lazy=lazy, **kwargs)

        # 2. Post-process (backend agnostic logic)
        ds = self._post_process(df, daily=daily)

        # 3. Add provenance
        ds = self.update_history(
            ds, "Loaded AirNow data via Aero Protocol modernized reader."
        )

        return ds

    def _post_process(self, df, daily=False):
        """Post-process the loaded DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame or dask.dataframe.DataFrame
        daily : bool

        Returns
        -------
        xarray.Dataset
        """
        from ..util import force_object_strings

        # Ensure object strings for consistency
        df = force_object_strings(df)

        # Convert to xarray (PointReader.to_xarray handles UGRID)
        # Note: we need to handle the time conversion first
        # We'll do it in a way that works for both Pandas and Dask

        def _fix_time(df_):
            df_ = df_.copy()
            if daily:
                df_["time"] = pd.to_datetime(
                    df_.date.astype(str), format=r"%m/%d/%y", errors="coerce"
                )
            else:
                df_["time"] = pd.to_datetime(
                    df_.date.astype(str) + " " + df_.time.astype(str),
                    format=r"%m/%d/%y %H:%M",
                    errors="coerce",
                )
            return df_.drop(columns=["date"], errors="ignore")

        if hasattr(df, "map_partitions"):
            meta = _fix_time(df._meta)
            df = df.map_partitions(_fix_time, meta=meta)
        else:
            df = _fix_time(df)

        ds = self.to_xarray(df, expand2d=False)

        # Add metadata if available (placeholder for now)
        # In a full implementation, we would merge with station metadata

        return ds
