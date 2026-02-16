import pytest
import pandas as pd
import xarray as xr
import numpy as np
from monetio.readers.aeronet import AERONETReader
from pathlib import Path

DATA = Path(__file__).parent / "data"

def test_aeronet_aero_protocol():
    # Use local file to avoid network issues
    fp = DATA / "aeronet-AOD15-example.txt"
    if not fp.exists():
        pytest.skip(f"Local data file not found at {fp}")

    reader = AERONETReader()

    # 1. Eager Load
    df_eager = reader.open_dataset(files=str(fp), as_xarray=False, lazy=False)
    assert isinstance(df_eager, pd.DataFrame)
    assert not df_eager.empty
    assert "time" in df_eager.columns
    assert "siteid" in df_eager.columns

    # 2. Lazy Load
    df_lazy = reader.open_dataset(files=str(fp), as_xarray=False, lazy=True)
    try:
        import dask.dataframe as dd
        assert isinstance(df_lazy, dd.DataFrame)
    except ImportError:
        pytest.skip("Dask not installed")

    # Check they match after compute
    # Sort columns as they might be in different order
    df_eager = df_eager.reindex(sorted(df_eager.columns), axis=1)
    df_lazy_computed = df_lazy.compute().reindex(sorted(df_eager.columns), axis=1)
    # Cast siteid to object to match dask's force_object_strings
    df_eager["siteid"] = df_eager["siteid"].astype(object)
    # Also time might have slightly different precision in display but should match
    pd.testing.assert_frame_equal(df_eager, df_lazy_computed)

    # 3. Xarray Eager
    ds_eager = reader.open_dataset(files=str(fp), as_xarray=True, lazy=False)
    assert isinstance(ds_eager, xr.Dataset)
    assert "time" in ds_eager.coords
    # In 2D expansion, siteid becomes 'node'
    assert "node" in ds_eager.dims

    # 4. Xarray Lazy
    ds_lazy = reader.open_dataset(files=str(fp), as_xarray=True, lazy=True)
    assert isinstance(ds_lazy, xr.Dataset)
    # Check if it's lazy (dask-backed)
    # For point data, the variables should be dask arrays
    assert any(ds_lazy[v].chunks is not None for v in ds_lazy.data_vars)

    # Match
    # Use check_like=True to ignore coordinate order
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())

def test_aeronet_build_urls():
    from monetio.readers.aeronet import build_urls
    dates = pd.date_range("2021-08-01", "2021-08-02", freq="D")
    urls = build_urls(dates, product="AOD15", siteid="SERC")
    assert len(urls) == 1
    assert "site=SERC" in urls[0]
    assert "AOD15=1" in urls[0]
    assert "year=2021&month=08&day=01" in urls[0]

    # Split by day
    urls_split = build_urls(dates, product="AOD15", siteid="SERC", split_by_day=True)
    assert len(urls_split) == 1 # 2021-08-01 to 2021-08-02 is one span

    dates_multi = pd.date_range("2021-08-01", "2021-08-03", freq="D")
    urls_multi = build_urls(dates_multi, split_by_day=True)
    assert len(urls_multi) == 2
