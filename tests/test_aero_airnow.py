import pandas as pd
import pytest
import xarray as xr

from monetio.readers.airnow import AirNowReader


def test_airnow_eager_vs_lazy():
    """
    Double-Check Test: Verify Eager (Pandas) and Lazy (Dask) results are identical.
    Following the Aero Protocol.
    """
    # Use a date that is likely to have data and be stable
    dates = pd.date_range("2024-07-01 00:00", periods=1, freq="h")

    # 1. Eager Load (NumPy/Pandas backend)
    # We use wide_fmt=False to ensure we are comparing the same long-format structure
    # since lazy=True currently stays in long format to avoid hidden computes.
    ds_eager = AirNowReader().open_dataset(dates=dates, as_xarray=True, lazy=False, wide_fmt=False)

    # 2. Lazy Load (Dask backend)
    ds_lazy = AirNowReader().open_dataset(dates=dates, as_xarray=True, lazy=True, wide_fmt=False)

    # Check that ds_lazy is indeed dask-backed
    # In PointReader.to_xarray, dask dataframes are converted to dask-backed DataArrays
    assert ds_lazy.obs.chunks is not None

    # Compute lazy result
    ds_lazy_computed = ds_lazy.compute()

    # Compare data - drop history as it contains timestamps
    ds_eager_no_hist = ds_eager.drop_vars("history", errors="ignore")
    ds_lazy_no_hist = ds_lazy_computed.drop_vars("history", errors="ignore")

    # Harmonize attributes for comparison (history is in attrs too sometimes)
    ds_eager_no_hist.attrs.pop("history", None)
    ds_lazy_no_hist.attrs.pop("history", None)

    xr.testing.assert_allclose(ds_eager_no_hist, ds_lazy_no_hist)

    # Verify provenance
    assert "history" in ds_eager.attrs
    assert "Read AirNow data" in ds_eager.attrs["history"]
    assert "Converted to xarray Dataset" in ds_eager.attrs["history"]


if __name__ == "__main__":
    pytest.main([__file__])
