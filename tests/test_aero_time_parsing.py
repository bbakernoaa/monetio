import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monetio.readers.sat_utils import tai93_to_datetime


def test_tai93_to_datetime_eager_lazy():
    """Test tai93_to_datetime with both Eager (NumPy) and Lazy (Dask) backends."""
    # 1. Setup sample data (seconds since 1993-01-01)
    # 0 -> 1993-01-01
    # 86400 -> 1993-01-02
    # 31536000 -> 1994-01-01
    times = np.array([0.0, 86400.0, 31536000.0])
    expected = pd.to_datetime(["1993-01-01", "1993-01-02", "1994-01-01"]).values.astype(
        "datetime64[ns]"
    )

    da_eager = xr.DataArray(times, dims="time", name="tai93")

    # 2. Test Eager
    res_eager = tai93_to_datetime(da_eager)
    assert isinstance(res_eager.data, np.ndarray)
    np.testing.assert_array_equal(res_eager.values, expected)

    # 3. Test Lazy (Dask)
    try:
        import dask.array  # noqa: F401

        da_lazy = da_eager.chunk({"time": 1})
        res_lazy = tai93_to_datetime(da_lazy)

        assert hasattr(res_lazy.data, "dask")
        # Verify result is identical to eager
        np.testing.assert_array_equal(res_lazy.compute().values, expected)
    except ImportError:
        pytest.skip("Dask not installed")


def test_tai93_to_datetime_precision():
    """Verify precision for fractional seconds."""
    # 0.5 seconds
    times = np.array([0.5])
    expected = np.datetime64("1993-01-01T00:00:00.500000000")

    da = xr.DataArray(times, dims="time")
    res = tai93_to_datetime(da)

    np.testing.assert_array_equal(res.values, [expected])


def test_tai93_to_datetime_2d():
    """Verify that 2D arrays (swaths) are handled correctly."""
    times = np.array([[0.0, 3600.0], [86400.0, 86400.0 + 3600.0]])
    expected = pd.to_datetime(
        ["1993-01-01T00:00:00", "1993-01-01T01:00:00", "1993-01-02T00:00:00", "1993-01-02T01:00:00"]
    ).values.astype("datetime64[ns]")

    da = xr.DataArray(times, dims=("y", "x"))
    res = tai93_to_datetime(da)

    assert res.shape == (2, 2)
    np.testing.assert_array_equal(res.values.ravel(), expected)
