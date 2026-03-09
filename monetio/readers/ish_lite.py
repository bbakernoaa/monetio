import os
from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from ..util import ds_to_2d, force_object_strings
from .core import PointReader
from .drivers import PandasDriver


def read_ish_lite_file(fname: str) -> pd.DataFrame:
    """Vectorized parsing of a single ISH Lite file.

    Parameters
    ----------
    fname : str
        Path to the ISH Lite file.

    Returns
    -------
    pd.DataFrame
        The parsed data.
    """
    columns = [
        "year",
        "month",
        "day",
        "hour",
        "temp",
        "dew_pt_temp",
        "press",
        "wdir",
        "ws",
        "sky_condition",
        "precip_1hr",
        "precip_6hr",
    ]
    # We use engine='python' or just the default. delim_whitespace=True is deprecated in newer pandas.
    # use sep='\s+' instead.
    df = pd.read_csv(
        fname,
        sep=r"\s+",
        header=None,
        names=columns,
    )

    # Manual time construction to avoid slow parse_dates
    # ISH Lite hours are 0-23
    df["time"] = pd.to_datetime(
        {
            "year": df["year"],
            "month": df["month"],
            "day": df["day"],
            "hour": df["hour"],
        }
    )

    # Extract siteid from filename (e.g., 722158-13897-2023.gz)
    basename = os.path.basename(fname)
    parts = basename.split("-")
    if len(parts) >= 2:
        siteid = parts[0] + parts[1]
    else:
        siteid = "unknown"
    df["siteid"] = siteid

    # Scaling and missing values
    for col in ["temp", "dew_pt_temp", "press", "ws", "precip_1hr", "precip_6hr"]:
        df[col] /= 10.0

    # Handle missing values as per legacy code
    df = df.replace(-999.9, np.nan)
    df = df.replace(-9999, np.nan)

    return df


class ISHLiteReader(PointReader):
    """Reader for NOAA ISH Lite (Integrated Surface Hourly) data."""

    def read_data(self, files: List[str], **kwargs: Any) -> pd.DataFrame:
        """Read data from ISH Lite files.

        Parameters
        ----------
        files : list of str
            List of file paths to read.

        Returns
        -------
        pd.DataFrame
            The aggregated data.
        """
        dfs = [read_ish_lite_file(f) for f in files]
        if not dfs:
            return pd.DataFrame()
        df = pd.concat(dfs, ignore_index=True)

        # Ensure strings are objects for Dask compatibility
        df = force_object_strings(df)

        return df

    def _post_process(
        self, ds: xr.Dataset, resample: bool = False, window: str = "h", **kwargs: Any
    ) -> xr.Dataset:
        """Post-process the dataset: resampling and history."""
        if resample:
            # Multi-site resampling requires 2D expansion first
            if ds.node.size > 0:
                # If not already 2D, expand it
                if "time" not in ds.dims or "node" not in ds.dims:
                    ds = ds_to_2d(ds)

                # Resample. Pandas 3.0 uses lowercase frequency codes.
                ds = ds.resample(time=window.lower()).mean()

        ds = self.update_history(ds, "Modernized ISH Lite reader via Aero Protocol")
        return ds


def open_dataset(
    files: List[str],
    dates: Optional[Sequence[Any]] = None,
    resample: bool = False,
    window: str = "h",
    **kwargs: Any,
) -> xr.Dataset:
    """Modernized open_dataset for ISH Lite.

    Parameters
    ----------
    files : list of str
        List of file paths to read.
    dates : sequence of datetime-like, optional
        Dates of interest.
    resample : bool, optional
        Whether to resample the data.
    window : str, optional
        Resampling window (e.g., 'h', '3h', 'D').
    **kwargs : dict
        Additional arguments (lazy, n_procs, expand2d, etc.).

    Returns
    -------
    xr.Dataset
        The loaded dataset.
    """
    reader = ISHLiteReader()
    driver = PandasDriver()

    # If resample is requested, we MUST expand2d to ensure per-site integrity
    expand2d = kwargs.get("expand2d", False) or resample

    ds = driver.open_dataset(files, reader, dates=dates, expand2d=expand2d, **kwargs)
    return reader._post_process(ds, resample=resample, window=window, **kwargs)
