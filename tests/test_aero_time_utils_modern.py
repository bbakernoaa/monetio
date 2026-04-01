import numpy as np
import xarray as xr

from monetio.readers.time_utils import parse_ioapi_times, parse_wrf_times, parse_yyyymmdd_hhmm


def test_parse_ioapi_times_aero():
    """Verify parse_ioapi_times works with both NumPy and Dask."""
    y = np.array([2023001, 2023002])
    h = np.array([120000, 183015])

    expected = np.array(["2023-01-01T12:00:00", "2023-01-02T18:30:15"], dtype="datetime64[ns]")

    # 1. Eager check
    res_eager = parse_ioapi_times(y, h)
    np.testing.assert_array_equal(res_eager, expected)

    # 2. Lazy check via xr.apply_ufunc
    dy = xr.DataArray(y, dims="time").chunk({"time": 1})
    dh = xr.DataArray(h, dims="time").chunk({"time": 1})

    res_lazy = xr.apply_ufunc(
        parse_ioapi_times, dy, dh, dask="parallelized", output_dtypes=[np.dtype("datetime64[ns]")]
    ).compute()

    np.testing.assert_array_equal(res_lazy.values, expected)


def test_parse_wrf_times_aero():
    """Verify parse_wrf_times works with both NumPy and Dask."""
    times = np.array(["2023-01-01_12:00:00", "2023-12-31_23:59:59"], dtype="S19")

    expected = np.array(["2023-01-01T12:00:00", "2023-12-31T23:59:59"], dtype="datetime64[ns]")

    # 1. Eager check
    res_eager = parse_wrf_times(times)
    np.testing.assert_array_equal(res_eager, expected)

    # 2. Lazy check
    dt = xr.DataArray(times, dims="time").chunk({"time": 1})

    res_lazy = xr.apply_ufunc(
        parse_wrf_times, dt, dask="parallelized", output_dtypes=[np.dtype("datetime64[ns]")]
    ).compute()

    np.testing.assert_array_equal(res_lazy.values, expected)


def test_parse_yyyymmdd_hhmm_aero():
    """Verify parse_yyyymmdd_hhmm works with both NumPy and Dask."""
    y = np.array([20230101, 20231231])
    h = np.array([1200, 2359])

    expected = np.array(["2023-01-01T12:00:00", "2023-12-31T23:59:00"], dtype="datetime64[ns]")

    # 1. Eager check
    res_eager = parse_yyyymmdd_hhmm(y, h)
    np.testing.assert_array_equal(res_eager, expected)

    # 2. Lazy check
    dy = xr.DataArray(y, dims="time").chunk({"time": 1})
    dh = xr.DataArray(h, dims="time").chunk({"time": 1})

    res_lazy = xr.apply_ufunc(
        parse_yyyymmdd_hhmm, dy, dh, dask="parallelized", output_dtypes=[np.dtype("datetime64[ns]")]
    ).compute()

    np.testing.assert_array_equal(res_lazy.values, expected)


def test_parse_yyyymmdd_hhmmss_aero():
    """Verify parse_yyyymmdd_hhmm works with HHMMSS format."""
    y = np.array([20230101])
    h = np.array([120005])

    expected = np.array(["2023-01-01T12:00:05"], dtype="datetime64[ns]")

    res = parse_yyyymmdd_hhmm(y, h)
    np.testing.assert_array_equal(res, expected)
