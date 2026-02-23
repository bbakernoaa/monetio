import numpy as np
import xarray as xr


def parse_ioapi_times(tflag: xr.DataArray) -> xr.DataArray:
    """Vectorized IOAPI time parser.

    Parameters
    ----------
    tflag : xr.DataArray
        TFLAG variable from IOAPI file. Expected shape (time, nvars, 2)
        or (time, 2).

    Returns
    -------
    xr.DataArray
        Parsed time coordinate with 'datetime64[ns]' dtype.
    """
    if tflag.ndim == 3:
        # Use first variable's tags (TSTEP, VAR, DATE-TIME)
        # We assume VAR is the 1st dimension and DATE-TIME is the 2nd
        # But to be safe, we use isel on the last two dimensions.
        if "VAR" in tflag.dims:
            tflag = tflag.isel(VAR=0)
        else:
            tflag = tflag.isel({tflag.dims[1]: 0})

    # Ensure we are working with the underlying data in a backend-agnostic way
    # xr.apply_ufunc will handle Dask vs NumPy
    return xr.apply_ufunc(
        _ioapi_to_dt,
        tflag[..., 0],
        tflag[..., 1],
        input_core_dims=[[], []],
        output_core_dims=[[]],
        dask="parallelized",
        output_dtypes=["datetime64[ns]"],
    )


def _ioapi_to_dt(yyyymmdd: np.ndarray, hhmmss: np.ndarray) -> np.ndarray:
    """Internal vectorized converter.

    Parameters
    ----------
    yyyymmdd : np.ndarray
        Array of Julian dates in YYYYDDD format.
    hhmmss : np.ndarray
        Array of times in HHMMSS format.

    Returns
    -------
    np.ndarray
        Array of datetime64[ns] objects.
    """
    # Handle both numpy and dask arrays (though apply_ufunc passes numpy chunks)
    years = (yyyymmdd // 1000).astype(int)
    days = (yyyymmdd % 1000).astype(int) - 1

    hours = (hhmmss // 10000).astype(int)
    minutes = ((hhmmss // 100) % 100).astype(int)
    seconds = (hhmmss % 100).astype(int)

    # Vectorized datetime arithmetic
    # Using numpy's datetime64 for speed and backend-agnostic behavior
    y_dt = (years - 1970).astype("datetime64[Y]").astype("datetime64[ns]")
    d_td = days.astype("timedelta64[D]").astype("timedelta64[ns]")
    h_td = hours.astype("timedelta64[h]").astype("timedelta64[ns]")
    m_td = minutes.astype("timedelta64[m]").astype("timedelta64[ns]")
    s_td = seconds.astype("timedelta64[s]").astype("timedelta64[ns]")

    return y_dt + d_td + h_td + m_td + s_td
