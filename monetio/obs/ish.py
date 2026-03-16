"""NOAA Integrated Surface Hourly (ISH; also known as ISD, Integrated Surface Data).

https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database
"""

import pandas as pd

from ..readers.ish import ISHReader


def add_data(
    dates,
    *,
    box=None,
    country=None,
    state=None,
    site=None,
    resample=True,
    window="h",
    download=False,
    n_procs=1,
    request_timeout=10,
    request_retries=4,
    verbose=False,
):
    """Retrieve and load ISH data as a DataFrame.

    Parameters
    ----------
    dates : sequence of datetime-like
    box : list of float, optional
            ``[latmin, lonmin, latmax, lonmax]``.
    country, state, site : str, optional
        Select sites in a country or state or one specific site.
        Can use one at most of `box` and these.
    resample : bool
        If false, return data at original resolution, which may be sub-hourly.
    window
        Resampling window, e.g. ``'3h'``.
    download : bool
        Ignored in the new implementation (S3/HTTPS accessed directly).
    n_procs : int
        For Dask.
    request_timeout : float
        Timeout (seconds) for requests.
    request_retries : int
        Number of retries.
    verbose : bool
        Print debugging messages.

    Returns
    -------
    DataFrame
    """
    reader = ISHReader()
    # open_dataset returns an xarray Dataset, we convert it to DataFrame for backward compatibility
    ds = reader.open_dataset(
        dates,
        site=site,
        state=state,
        country=country,
        box=box,
        resample=resample,
        window=window,
        lazy=False,  # add_data historically returns a pandas DataFrame immediately
        request_timeout=request_timeout,
        request_retries=request_retries,
        verbose=verbose,
    )

    if ds.sizes == {}:
        return pd.DataFrame()

    df = ds.to_dataframe().reset_index()

    # Restore legacy column names if necessary
    # The new reader uses 'siteid', legacy uses 'usaf' and 'wban'
    if "siteid" in df.columns:
        df["usaf"] = df["siteid"].str[:6]
        df["wban"] = df["siteid"].str[6:]

    return df


class ISH:
    """Legacy ISH class for backward compatibility."""

    def __init__(self):
        self.reader = ISHReader()
        self.history = None
        self.dates = None

    def read_ish_history(self, dates=None):
        self.history = self.reader.read_history()
        if dates is not None:
            self.history = self.history.loc[
                (self.history.end >= dates.min()) & (self.history.begin <= dates.max())
            ]

    def add_data(self, *args, **kwargs):
        return add_data(*args, **kwargs)
