import numpy as np
import pandas as pd
import xarray as xr


def parse_ioapi_times(date, time):
    """Vectorized IOAPI time parser (Julian YYYYDDD, HHMMSS).

    Parameters
    ----------
    date : xarray.DataArray
        Julian date (YYYYDDD).
    time : xarray.DataArray
        Time (HHMMSS).

    Returns
    -------
    xarray.DataArray
        Parsed datetime64[ns].
    """
    # Vectorized logic:
    # 1. Convert Julian date to YYYY-MM-DD
    # 2. Convert HHMMSS to HH:MM:SS
    # 3. Combine and parse

    def _parse(d, t):
        # Implementation for a single element or NumPy array
        # d: YYYYDDD, t: HHMMSS
        # Use vectorized arithmetic to avoid Python loops
        y = d // 1000
        days = (d % 1000) - 1
        h = t // 10000
        m = (t // 100) % 100
        s = t % 100

        # Create datetime64[ns]
        res = (
            np.datetime64("1970-01-01")
            + np.array(y - 1970, dtype="timedelta64[Y]")
            + np.array(days, dtype="timedelta64[D]")
            + np.array(h, dtype="timedelta64[h]")
            + np.array(m, dtype="timedelta64[m]")
            + np.array(s, dtype="timedelta64[s]")
        )
        return res.astype("datetime64[ns]")

    return xr.apply_ufunc(
        _parse,
        date,
        time,
        dask="parallelized",
        output_dtypes=["datetime64[ns]"],
    )


def parse_wrf_times(times):
    """Vectorized WRF time parser (character array strings).

    Parameters
    ----------
    times : xarray.DataArray
        Times character array.

    Returns
    -------
    xarray.DataArray
        Parsed datetime64[ns].
    """

    def _parse(t):
        # t: character array (e.g., |S19)
        # Convert to string and parse
        return pd.to_datetime(t.astype(str), format="%Y-%m-%d_%H:%M:%S").values.astype(
            "datetime64[ns]"
        )

    return xr.apply_ufunc(
        _parse,
        times,
        dask="parallelized",
        output_dtypes=["datetime64[ns]"],
    )
