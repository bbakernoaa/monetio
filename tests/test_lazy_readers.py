import numpy as np
import pandas as pd
import pytest

import monetio


@pytest.mark.parametrize("reader_name", ["airnow", "ish_lite"])
def test_lazy_consistency(reader_name):
    """Verify that lazy=True and lazy=False produce consistent data."""

    # Use a small range for speed and to avoid large downloads
    if reader_name == "airnow":
        dates = pd.date_range("2024-07-01", periods=2, freq="h")
        # AirNow stays lazy ONLY if wide_fmt=False
        kwargs = {"wide_fmt": False}
    elif reader_name == "ish_lite":
        dates = pd.date_range("2020-09-01", periods=1, freq="h")
        kwargs = {"site": "72224400358", "resample": False}  # College Park
    else:
        return

    # 1. Eager (NumPy/Pandas)
    ds_eager = monetio.load(reader_name, dates=dates, as_xarray=True, lazy=False, **kwargs)

    # 2. Lazy (Dask)
    ds_lazy = monetio.load(reader_name, dates=dates, as_xarray=True, lazy=True, **kwargs)

    # Check that ds_lazy is actually dask-backed
    varname = [v for v in ds_lazy.data_vars if v != "mesh"][0]
    assert hasattr(ds_lazy[varname].data, "dask")

    # Compute lazy
    ds_lazy_computed = ds_lazy.compute()

    # Compare values
    for var in ds_eager.data_vars:
        if var == "mesh":
            continue
        v1 = ds_eager[var].values.ravel()
        v2 = ds_lazy_computed[var].values.ravel()

        # Remove NaNs for comparison
        # Handle object arrays carefully for NaNs
        if v1.dtype == object:
            mask1 = pd.notna(v1)
            mask2 = pd.notna(v2)
        else:
            mask1 = ~np.isnan(v1)
            mask2 = ~np.isnan(v2)

        v1 = v1[mask1]
        v2 = v2[mask2]

        if v1.dtype.kind in "ifc":  # numeric
            np.testing.assert_allclose(np.sort(v1), np.sort(v2), atol=1e-5)
        else:
            np.testing.assert_array_equal(np.sort(v1.astype(str)), np.sort(v2.astype(str)))


def test_airnow_history():
    """Verify provenance tracking."""
    dates = pd.date_range("2024-07-01", periods=1, freq="h")
    ds = monetio.load("airnow", dates=dates, as_xarray=True)
    assert "history" in ds.attrs
    assert "Read AirNow data" in ds.attrs["history"]
    assert "Converted to xarray Dataset" in ds.attrs["history"]
