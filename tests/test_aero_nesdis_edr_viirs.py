import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monetio.readers.nesdis_edr_viirs import NESDISEDRVIIRSReader


@pytest.fixture
def dummy_binary_file(tmp_path):
    """Create a dummy binary file for NESDIS EDR VIIRS."""
    fname = tmp_path / "npp_aot550_edr_gridded_0.25_20230101.high.bin"
    nlat, nlon = 720, 1440
    # 2 layers: AOD and something else
    data = np.random.rand(2, nlat, nlon).astype(np.float32)
    # Add some invalid values to test masking
    data[0, 0, 0] = -1000.0
    data.tofile(fname)
    return str(fname)


def test_nesdis_edr_viirs_eager_lazy_consistency(dummy_binary_file):
    """Verify that Eager and Lazy modes return identical results."""
    reader = NESDISEDRVIIRSReader()
    date = pd.Timestamp("2023-01-01")

    # Eager (NumPy)
    ds_eager = reader.read_data(dummy_binary_file, date, resolution="low", lazy=False)

    # Lazy (Dask)
    ds_lazy = reader.read_data(dummy_binary_file, date, resolution="low", lazy=True)

    # Assertions
    # 1. Check data values (after computing lazy)
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())

    # 2. Check types
    assert isinstance(ds_eager.aod_550.data, np.ndarray)
    import dask.array as da

    assert isinstance(ds_lazy.aod_550.data, da.Array)

    # 3. Check coordinates
    assert "latitude" in ds_lazy.coords
    assert "longitude" in ds_lazy.coords
    assert "time" in ds_lazy.coords
    assert ds_lazy.latitude.ndim == 2
    assert ds_lazy.longitude.ndim == 2

    # 4. Check masking
    assert np.isnan(ds_eager.aod_550.isel(time=0, y=0, x=0))
    assert np.isnan(ds_lazy.aod_550.isel(time=0, y=0, x=0).compute())


def test_nesdis_edr_viirs_metadata(dummy_binary_file):
    """Check metadata and attributes."""
    reader = NESDISEDRVIIRSReader()
    date = pd.Timestamp("2023-01-01")
    ds = reader.read_data(dummy_binary_file, date, resolution="low", lazy=False)

    assert ds.aod_550.attrs["units"] == "1"
    assert "history" in ds.attrs
    assert "Aero Protocol" in ds.attrs["history"]
    assert ds.latitude.attrs["units"] == "degrees_north"
    assert ds.longitude.attrs["units"] == "degrees_east"
