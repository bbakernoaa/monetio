import numpy as np
import pytest
import xarray as xr
import pandas as pd
from monetio.readers.base import _ensure_time_dimension
from monetio.readers.madis import MADISReader
from monetio.readers.nesdis_frp import nesdis_frp_preprocess

def test_ensure_time_dimension_lazy():
    da = xr.DataArray(np.random.rand(3), dims="x", name="test")
    time_val = np.datetime64("2023-01-01")

    # Eager case
    ds_eager = xr.Dataset({"a": da}, coords={"time": time_val})
    ds_eager_out = _ensure_time_dimension(ds_eager)
    assert "time" in ds_eager_out.dims
    assert ds_eager_out.time.size == 1
    assert ds_eager_out.time.values[0] == time_val

    # Lazy case
    try:
        import dask.array as da_lazy
        # Scalar dask array for time
        time_lazy = da_lazy.from_array(time_val)
        ds_lazy = xr.Dataset(
            {"a": (("x",), da_lazy.from_array(np.random.rand(3), chunks=2))},
            coords={"time": xr.DataArray(time_lazy, coords={}, dims=[])}
        )

        # Check that it is indeed lazy
        assert hasattr(ds_lazy.time.data, "dask")

        ds_lazy_out = _ensure_time_dimension(ds_lazy)

        assert "time" in ds_lazy_out.dims
        assert ds_lazy_out.time.size == 1

        # NOTE: Xarray often computes index coordinates immediately.
        # But data variables should remain lazy.
        assert hasattr(ds_lazy_out.a.data, "dask")

        # Verify result is correct when computed
        assert ds_lazy_out.time.values[0] == time_val
    except ImportError:
        pytest.skip("Dask not installed")

def test_madis_harmonize_lazy():
    try:
        import dask.array as da_lazy
        # Epoch seconds for 2023-01-01 00:00:00
        seconds = 1672531200
        time_data = da_lazy.from_array([seconds], chunks=1)

        ds = xr.Dataset(
            {"time": (("node",), time_data)},
            attrs={"units": "seconds since 1970-01-01 00:00:00.0 +0000"}
        )
        ds["time"].attrs["units"] = "seconds since 1970-01-01 00:00:00.0 +0000"

        reader = MADISReader()
        ds_out = reader.harmonize(ds)

        # Should still be lazy
        assert hasattr(ds_out.time.data, "dask")

        # Should be converted to datetime64
        assert ds_out.time.dtype.kind == "M"
        assert ds_out.time.compute()[0] == np.datetime64("2023-01-01")
    except ImportError:
        pytest.skip("Dask not installed")

def test_nesdis_frp_preprocess_lazy():
    try:
        import dask.array as da_lazy
        tile_data = da_lazy.from_array([1], chunks=1)
        ds = xr.Dataset(
            {"frp": (("x", "y"), da_lazy.from_array(np.random.rand(2, 2), chunks=1))},
            coords={"tile": ((), tile_data[0])}
        )

        # Preprocess
        ds_out = nesdis_frp_preprocess(ds, ftype="meanFRP")

        # FRP should still be lazy
        assert hasattr(ds_out.meanFRP.data, "dask")
        # Tile should still be lazy (we avoided .values)
        assert hasattr(ds_out.tile.data, "dask")

    except ImportError:
        pytest.skip("Dask not installed")
