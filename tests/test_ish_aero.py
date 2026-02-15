import pandas as pd
import pytest
import xarray as xr
from monetio.readers.ish import ISHReader

def test_ish_eager_vs_lazy():
    """Verify that Eager and Lazy loading of ISH data produces identical results."""
    # Use a single day to keep it fast
    dates = pd.date_range("2020-09-01", "2020-09-01 23:00", freq="h")
    site = "72224400358" # College Park AP

    reader = ISHReader()

    # 1. Eager Load (as_xarray=False)
    df_eager = reader.open_dataset(dates, site=site, as_xarray=False, lazy=False, resample=False)

    # 2. Lazy Load (as_xarray=False)
    df_lazy = reader.open_dataset(dates, site=site, as_xarray=False, lazy=True, resample=False)

    # Check that df_lazy is indeed lazy
    import dask.dataframe as dd
    assert isinstance(df_lazy, dd.DataFrame)

    # Compute
    df_lazy_c = df_lazy.compute()

    # Compare DataFrames
    # Sort for consistency
    df_eager = df_eager.sort_values(["time", "siteid"]).reset_index(drop=True)
    df_lazy_c = df_lazy_c.sort_values(["time", "siteid"]).reset_index(drop=True)

    # Handle dtypes (Dask might have object instead of string for some columns)
    for col in df_eager.columns:
        if df_eager[col].dtype != df_lazy_c[col].dtype:
             df_lazy_c[col] = df_lazy_c[col].astype(df_eager[col].dtype)

    pd.testing.assert_frame_equal(df_eager, df_lazy_c)

def test_ish_lazy_resample():
    """Verify that resampling works on a lazy ISH dataset."""
    # Use a small range to avoid timeout
    dates = pd.date_range("2020-09-01", "2020-09-01 23:00", freq="h")
    site = "72224400358"

    reader = ISHReader()

    # Load lazy with resample=True
    ds = reader.open_dataset(dates, site=site, as_xarray=True, lazy=True, resample=True, window="3h")

    # Compute
    ds_c = ds.compute()

    # Check time frequency
    assert "time" in ds_c.dims

    time_diffs = ds_c.time.diff("time")
    assert (time_diffs == pd.Timedelta("3h")).all()
    # 24 hours / 3 = 8
    assert len(ds_c.time) == 8
