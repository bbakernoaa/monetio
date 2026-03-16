"""Time parsing utilities."""

import numpy as np
import pandas as pd


def parse_ioapi_times(yyyymmdd: np.ndarray, hhmmss: np.ndarray) -> np.ndarray:
    """
    Vectorized parsing of IOAPI (CMAQ/CAMx) TFLAG dates and times.

    Parameters
    ----------
    yyyymmdd : np.ndarray
        Array of dates in YYYYDDD format (Julian day).
    hhmmss : np.ndarray
        Array of times in HHMMSS format.

    Returns
    -------
    np.ndarray
        Array of datetime64[ns] objects.
    """
    # 1. Extract components using math
    years = (yyyymmdd // 1000).astype(int)
    days = (yyyymmdd % 1000).astype(int)

    hours = (hhmmss // 10000).astype(int)
    minutes = ((hhmmss // 100) % 100).astype(int)
    seconds = (hhmmss % 100).astype(int)

    # 2. Convert to nanoseconds since epoch
    # We use pandas to handle the variable year starts (leap years)
    # This part is vectorized and fast.
    # We use np.unique to avoid repeated parsing of the same year
    unique_years = np.unique(years)
    year_to_start = {y: pd.to_datetime(str(y), format="%Y").to_datetime64() for y in unique_years}

    year_starts = np.array([year_to_start[y] for y in years], dtype="datetime64[ns]")

    # 3. Combine using datetime arithmetic
    # Julian day 1 is the first day of the year (0 offset)
    day_offset = (days - 1).astype("timedelta64[D]")
    time_offset = (
        hours.astype("timedelta64[h]")
        + minutes.astype("timedelta64[m]")
        + seconds.astype("timedelta64[s]")
    )

    # Force ns to match project expectations and avoid discrepancies with Pandas 3.0+
    return (year_starts + day_offset + time_offset).astype("datetime64[ns]")


def parse_wrf_times(times_arr: np.ndarray) -> np.ndarray:
    """
    Vectorized parsing of WRF/RAQMS character array or string times.

    Parameters
    ----------
    times_arr : np.ndarray
        Array of times as strings or character arrays.

    Returns
    -------
    np.ndarray
        Array of datetime64[ns] objects.
    """
    if times_arr.ndim > 1:
        # It's likely a character array (..., DateStrLen)
        last_dim = times_arr.shape[-1]
        if times_arr.dtype.kind == "U":
            # Unicode: join along the last axis
            orig_shape = times_arr.shape
            flat_times = times_arr.reshape(-1, last_dim)
            s = np.array(["".join(row) for row in flat_times])
            s = s.reshape(orig_shape[:-1])
        else:
            # Bytes: use view if C-contiguous
            if times_arr.flags.c_contiguous:
                s = times_arr.view(f"S{last_dim}").squeeze(-1)
            else:
                orig_shape = times_arr.shape
                flat_times = times_arr.reshape(-1, last_dim)
                s = np.array([b"".join(row) for row in flat_times])
                s = s.reshape(orig_shape[:-1])
    else:
        s = times_arr

    # Replace '_' with ' ' for pandas compatibility (WRF format: YYYY-MM-DD_HH:MM:SS)
    if s.dtype.kind in {"S", "a"}:
        s = np.char.replace(s, b"_", b" ")
        s = s.astype(str)
    else:
        s = np.char.replace(s.astype(str), "_", " ")

    # Force ns to match project expectations and avoid discrepancies with Pandas 3.0+
    return pd.to_datetime(s.ravel()).values.astype("datetime64[ns]").reshape(s.shape)


def parse_yyyymmdd_hhmm(yyyymmdd: np.ndarray, hhmm: np.ndarray) -> np.ndarray:
    """
    Vectorized parsing of YYYYMMDD and HHMM (or HHMMSS) dates and times.

    Parameters
    ----------
    yyyymmdd : np.ndarray
        Array of dates in YYYYMMDD format.
    hhmm : np.ndarray
        Array of times in HHMM or HHMMSS format.

    Returns
    -------
    np.ndarray
        Array of datetime64[ns] objects.
    """
    yyyymmdd = np.asanyarray(yyyymmdd)
    hhmm = np.asanyarray(hhmm)

    # Use float for intermediate to handle NaNs if present
    years = (yyyymmdd // 10000).astype(float)
    months = ((yyyymmdd // 100) % 100).astype(float)
    days = (yyyymmdd % 100).astype(float)

    # HHMMSS check: 2400 is the threshold (HHMM max is 2359)
    # We use nanmax safely.
    try:
        is_hhmmss = (
            np.nanmax(hhmm) >= 10000 if hhmm.size > 0 and not np.all(np.isnan(hhmm)) else False
        )
    except (ValueError, TypeError):
        is_hhmmss = False

    if is_hhmmss:
        hours = (hhmm // 10000).astype(float)
        minutes = ((hhmm // 100) % 100).astype(float)
        seconds = (hhmm % 100).astype(float)
    else:
        hours = (hhmm // 100).astype(float)
        minutes = (hhmm % 100).astype(float)
        seconds = 0.0

    df = pd.DataFrame(
        {
            "year": years.ravel(),
            "month": months.ravel(),
            "day": days.ravel(),
            "hour": hours.ravel(),
            "minute": minutes.ravel(),
            "second": seconds if isinstance(seconds, float) else seconds.ravel(),
        }
    )

    # Use coerce to handle any invalid dates produced by math on NaNs
    return (
        pd.to_datetime(df, errors="coerce").values.astype("datetime64[ns]").reshape(yyyymmdd.shape)
    )
