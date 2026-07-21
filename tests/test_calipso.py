"""
Test CALIPSO Reader
"""

import numpy as np
import pytest
import xarray as xr

from monetio.readers.calipso import CALIPSOReader


def create_mock_calipso_dataset(is_dask=False):
    """Create a mock CALIOP L2 aerosol profile dataset."""
    nray = 10
    nbin = 30

    ds = xr.Dataset(
        data_vars={
            "Extinction_Coefficient_532": (("nray", "nbin"), np.random.rand(nray, nbin)),
            "Latitude": (("nray",), np.linspace(-90, 90, nray)),
            "Longitude": (("nray",), np.linspace(-180, 180, nray)),
            "Profile_Time": (("nray",), np.linspace(0, 6000, nray)),
        }
    )

    if is_dask:
        ds = ds.chunk({"nray": 5, "nbin": 15})

    return ds


@pytest.mark.parametrize("lazy", [False, True])
def test_calipso_reader_logic(lazy, monkeypatch):
    """Test CALIPSO reader preprocessing logic."""
    mock_ds = create_mock_calipso_dataset(is_dask=lazy)

    def mock_open(*args, **kwargs):
        return mock_ds

    monkeypatch.setattr("monetio.readers.drivers.XarrayDriver.open", mock_open)

    reader = CALIPSOReader()
    ds = reader.open_dataset(files="dummy.hdf")

    assert "latitude" in ds.coords
    assert "longitude" in ds.coords
    assert "time" in ds.coords
    assert ds.dims == {"y": 10, "z": 30}

    # Verify time is converted to standard datetime64[ns]
    assert np.issubdtype(ds.coords["time"].dtype, np.datetime64)
    # 0 seconds since 1993-01-01 should be 1993-01-01T00:00:00
    assert ds.coords["time"].values[0] == np.datetime64("1993-01-01T00:00:00")

    if lazy:
        assert ds.Extinction_Coefficient_532.chunks is not None
        assert ds.coords["time"].chunks is not None
