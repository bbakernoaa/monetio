import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monetio.readers.time_utils import parse_ioapi_times, parse_wrf_times


def test_parse_ioapi_times_eager():
    """Test IOAPI time parsing with NumPy arrays."""
    yyyymmdd = np.array([2023001, 2023001, 2023365, 2024001], dtype=int)
    hhmmss = np.array([0, 120000, 235959, 100], dtype=int)

    expected = pd.to_datetime(
        ["2023-01-01 00:00:00", "2023-01-01 12:00:00", "2023-12-31 23:59:59", "2024-01-01 00:01:00"]
    ).values.astype("datetime64[ns]")

    result = parse_ioapi_times(yyyymmdd, hhmmss)
    np.testing.assert_array_equal(result, expected)


def test_parse_ioapi_times_lazy():
    """Test IOAPI time parsing with Dask via xarray."""
    pytest.importorskip("dask.array")

    yyyymmdd = xr.DataArray(np.array([2023001, 2023365], dtype=int), dims="time").chunk(1)
    hhmmss = xr.DataArray(np.array([120000, 235959], dtype=int), dims="time").chunk(1)

    result = xr.apply_ufunc(
        parse_ioapi_times,
        yyyymmdd,
        hhmmss,
        dask="parallelized",
        output_dtypes=[np.dtype("datetime64[ns]")],
    )

    expected = pd.to_datetime(["2023-01-01 12:00:00", "2023-12-31 23:59:59"]).values.astype(
        "datetime64[ns]"
    )

    np.testing.assert_array_equal(result.compute().values, expected)


def test_parse_wrf_times_char_array():
    """Test WRF time parsing with character arrays (NumPy)."""
    # WRF format is often (time, 19)
    times_list = ["2023-10-27_00:00:00", "2023-10-27_12:30:05"]
    char_arr = np.array([list(s) for s in times_list], dtype="S1")

    expected = pd.to_datetime(["2023-10-27 00:00:00", "2023-10-27 12:30:05"]).values.astype(
        "datetime64[ns]"
    )

    result = parse_wrf_times(char_arr)
    np.testing.assert_array_equal(result, expected)


def test_parse_wrf_times_string_array():
    """Test WRF time parsing with string arrays (NumPy)."""
    s_arr = np.array(["2023-10-27_00:00:00", "2023-10-27_12:30:05"], dtype="S19")

    expected = pd.to_datetime(["2023-10-27 00:00:00", "2023-10-27 12:30:05"]).values.astype(
        "datetime64[ns]"
    )

    result = parse_wrf_times(s_arr)
    np.testing.assert_array_equal(result, expected)


def test_parse_wrf_times_lazy():
    """Test WRF time parsing with Dask via xarray."""
    pytest.importorskip("dask.array")

    times_list = ["2023-10-27_00:00:00", "2023-10-27_12:30:05"]
    char_arr = np.array([list(s) for s in times_list], dtype="S1")
    da = xr.DataArray(char_arr, dims=("time", "DateStrLen")).chunk({"time": 1})

    result = xr.apply_ufunc(
        parse_wrf_times,
        da,
        input_core_dims=[["DateStrLen"]],
        output_core_dims=[[]],
        dask="parallelized",
        output_dtypes=[np.dtype("datetime64[ns]")],
    )

    expected = pd.to_datetime(["2023-10-27 00:00:00", "2023-10-27 12:30:05"]).values.astype(
        "datetime64[ns]"
    )

    np.testing.assert_array_equal(result.compute().values, expected)
