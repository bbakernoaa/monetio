import numpy as np
import pandas as pd
import pytest
import xarray as xr
from monetio.readers.nesdis_eps_viirs import nesdis_eps_viirs_preprocess

def make_mock_ds(seed=42):
    """Create a mock NESDIS EPS VIIRS dataset."""
    nlat = 10
    nlon = 20

    rng = np.random.default_rng(seed)
    # Mock data variable with a dummy time dimension to match expand_dims in preprocess
    # Actually, preprocess adds time dimension if not present.
    # The original data in the file is (y, x)
    data = rng.standard_normal((nlat, nlon)).astype(np.float32)
    data[0, 0] = -1.0  # Should be masked

    ds = xr.Dataset(
        data_vars={
            "aot_ip_out": (("y", "x"), data),
        },
        attrs={
            "time_coverage_start": "2023-01-01T00:00:00Z",
        }
    )
    return ds

def test_nesdis_eps_viirs_eager_lazy_consistency():
    """Verify Eager (NumPy) and Lazy (Dask) consistency for NESDIS EPS VIIRS reader."""
    ds_eager = make_mock_ds()

    # 1. Process Eager
    ds_eager_proc = nesdis_eps_viirs_preprocess(ds_eager)

    # 2. Process Lazy
    ds_lazy = make_mock_ds().chunk({"y": 5, "x": 5})
    ds_lazy_proc = nesdis_eps_viirs_preprocess(ds_lazy)

    # Verify laziness
    assert hasattr(ds_lazy_proc.aod_550.data, "dask")
    assert hasattr(ds_lazy_proc.latitude.data, "dask")
    assert hasattr(ds_lazy_proc.longitude.data, "dask")

    # 3. Compare Results
    # Compute lazy result
    ds_lazy_computed = ds_lazy_proc.compute()

    # Assert coordinates are identical
    xr.testing.assert_allclose(ds_eager_proc.latitude, ds_lazy_computed.latitude)
    xr.testing.assert_allclose(ds_eager_proc.longitude, ds_lazy_computed.longitude)

    # Assert data is identical (including masking)
    xr.testing.assert_allclose(ds_eager_proc.aod_550, ds_lazy_computed.aod_550)

    # Assert masking worked (-1.0 -> NaN)
    # Note: aod_550 becomes (time, y, x) after preprocess
    val = ds_eager_proc.aod_550.values[0, 0, 0]
    assert np.isnan(val)

    # Assert Time
    assert ds_eager_proc.time.values == ds_lazy_computed.time.values

    # Assert History
    assert "history" in ds_eager_proc.attrs
    assert "Preprocessed NESDIS EPS VIIRS data" in ds_eager_proc.attrs["history"]
    assert "history" in ds_lazy_proc.attrs
    assert "Preprocessed NESDIS EPS VIIRS data" in ds_lazy_proc.attrs["history"]

if __name__ == "__main__":
    pytest.main([__file__])
