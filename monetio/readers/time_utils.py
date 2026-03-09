import pandas as pd
import xarray as xr


def parse_ioapi_times(dates: xr.DataArray, times: xr.DataArray) -> xr.DataArray:
    """Vectorized time parser for IOAPI Julian YYYYDDD/HHMMSS.

    Parameters
    ----------
    dates : xr.DataArray
        The dates in YYYYDDD format.
    times : xr.DataArray
        The times in HHMMSS format.

    Returns
    -------
    xr.DataArray
        The parsed times as datetime64[ns].
    """
    import numpy as np

    def _parse(d, t):
        y = d // 1000
        d_ = (d % 1000) - 1
        h = t // 10000
        m = (t // 100) % 100
        s = t % 100
        res = (
            (y - 1970).astype("datetime64[Y]")
            + d_.astype("timedelta64[D]")
            + h.astype("timedelta64[h]")
            + m.astype("timedelta64[m]")
            + s.astype("timedelta64[s]")
        )
        return res.astype("datetime64[ns]")

    return xr.apply_ufunc(
        _parse, dates, times, dask="parallelized", output_dtypes=[np.dtype("datetime64[ns]")]
    )


def parse_wrf_times(times: xr.DataArray) -> xr.DataArray:
    """Vectorized time parser for WRF character-array strings.

    Parameters
    ----------
    times : xr.DataArray
        The character-array strings of times.

    Returns
    -------
    xr.DataArray
        The parsed times as datetime64[ns].
    """
    import numpy as np

    def _parse(t):
        if t.dtype.kind == "S":
            t = t.astype("U")
        return pd.to_datetime(t, format="%Y-%m-%d_%H:%M:%S").values.astype("datetime64[ns]")

    return xr.apply_ufunc(
        _parse,
        times,
        dask="parallelized",
        vectorize=True,
        output_dtypes=[np.dtype("datetime64[ns]")],
    )
