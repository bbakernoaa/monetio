from typing import Union

import numpy as np
import pandas as pd
import xarray as xr


def _vectorized_parse_yyyymmdd_hhmm(date_arr: np.ndarray, time_arr: np.ndarray) -> np.ndarray:
    """Internal NumPy-only vectorized parser."""
    d_str = date_arr.astype(str)
    # Handle different lengths of time strings by padding
    t_str = np.char.zfill(time_arr.astype(str), 4)

    # YYYYMMDD -> YYYY-MM-DD
    year = np.char.slice_(d_str, 0, 4)
    month = np.char.slice_(d_str, 4, 6)
    day = np.char.slice_(d_str, 6, 8)

    # HHMM -> HH:MM
    hour = np.char.slice_(t_str, 0, 2)
    minute = np.char.slice_(t_str, 2, 4)

    # Construct ISO 8601 strings: YYYY-MM-DDTHH:MM
    iso = np.char.add(year, "-")
    iso = np.char.add(iso, month)
    iso = np.char.add(iso, "-")
    iso = np.char.add(iso, day)
    iso = np.char.add(iso, "T")
    iso = np.char.add(iso, hour)
    iso = np.char.add(iso, ":")
    iso = np.char.add(iso, minute)

    return iso.astype("datetime64[ns]")


def parse_yyyymmdd_hhmm(
    date_arr: Union[xr.DataArray, np.ndarray, pd.Series],
    time_arr: Union[xr.DataArray, np.ndarray, pd.Series],
) -> Union[xr.DataArray, np.ndarray, pd.Series]:
    """Vectorized parsing of YYYYMMDD and HHMM strings or integers into datetime64[ns].

    Supports Dask-backed Xarray objects.

    Parameters
    ----------
    date_arr : Union[xr.DataArray, np.ndarray, pd.Series]
        Array of dates in YYYYMMDD format.
    time_arr : Union[xr.DataArray, np.ndarray, pd.Series]
        Array of times in HHMM format.

    Returns
    -------
    Union[xr.DataArray, np.ndarray, pd.Series]
        Parsed datetimes.
    """
    if isinstance(date_arr, xr.DataArray) and isinstance(time_arr, xr.DataArray):
        return xr.apply_ufunc(
            _vectorized_parse_yyyymmdd_hhmm,
            date_arr,
            time_arr,
            dask="parallelized",
            output_dtypes=["datetime64[ns]"],
        )
    elif isinstance(date_arr, (pd.Series, pd.Index)):
        # For pandas, we can just use the internal NumPy function directly on values
        res = _vectorized_parse_yyyymmdd_hhmm(date_arr.values, time_arr.values)
        return pd.Series(res, index=date_arr.index)
    else:
        return _vectorized_parse_yyyymmdd_hhmm(np.asarray(date_arr), np.asarray(time_arr))
