"""NOAA Integrated Surface Hourly (ISH; also known as ISD, Integrated Surface Data) lite version.

https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/isd-lite-format.txt
"""

import pandas as pd

from ..readers.ish import ISHLiteReader


def add_data(
    dates,
    *,
    box=None,
    country=None,
    state=None,
    site=None,
    resample=False,
    window="h",
    n_procs=1,
    verbose=False,
):
    """Retrieve and load ISH-lite data as a DataFrame.

    Parameters
    ----------
    dates : sequence of datetime-like
    box : list of float, optional
            ``[latmin, lonmin, latmax, lonmax]``.
    country, state, site : str, optional
        Select sites in a country or state or one specific site.
        Can use one at most of `box` and these.
    resample : bool
    window
        Resampling window, e.g. ``'3h'``.
    n_procs : int
        For Dask.
    verbose : bool
        Print debugging messages.

    Returns
    -------
    DataFrame
    """
    reader = ISHLiteReader()
    ds = reader.open_dataset(
        dates,
        site=site,
        state=state,
        country=country,
        box=box,
        resample=resample,
        window=window,
        lazy=False,
        verbose=verbose,
    )

    if ds.sizes == {}:
        return pd.DataFrame()

    return ds.to_dataframe().reset_index()


class ISH:
    """Legacy ISH Lite class for backward compatibility."""

    def __init__(self):
        self.reader = ISHLiteReader()
        self.history = None

    def read_ish_history(self, dates=None):
        self.history = self.reader.read_history()

    def add_data(self, *args, **kwargs):
        return add_data(*args, **kwargs)
