import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monetio.readers.icap_mme import ICAPMMEReader


def create_mock_icap_ds(lazy=False):
    """Creates a mock ICAP-MME dataset."""
    nt, nlat, nlon = 4, 10, 20

    # Use fixed data
    data = {
        "dustaod550": (
            ("time", "lat", "lon"),
            np.arange(nt * nlat * nlon).reshape(nt, nlat, nlon).astype(float),
        ),
    }

    ds = xr.Dataset(
        data_vars=data,
        coords={
            "time": (("time",), pd.date_range("2023-01-01", periods=nt, freq="6h")),
            "lat": (("lat",), np.linspace(-90, 90, nlat)),
            "lon": (("lon",), np.linspace(-180, 180, nlon)),
        },
    )

    if lazy:
        ds = ds.chunk({"time": 2, "lat": -1, "lon": -1})

    return ds


def test_icap_protocol_compliance():
    """Verify ICAP processing is backend-agnostic and lazy-friendly."""
    ds_base = create_mock_icap_ds(lazy=False)
    ds_eager = ds_base.copy(deep=True)
    ds_lazy = ds_base.chunk({"time": 2, "lat": -1, "lon": -1})

    reader = ICAPMMEReader()

    class MockDriver:
        def __init__(self, ds):
            self.ds = ds

        def open(self, *args, **kwargs):
            return self.ds

    # Test Eager
    reader.driver = MockDriver(ds_eager)
    res_eager = reader.open_dataset(files="dummy.nc")

    # Test Lazy
    reader.driver = MockDriver(ds_lazy)
    res_lazy = reader.open_dataset(files="dummy.nc", lazy=True)

    # Check laziness
    assert isinstance(res_lazy.dustaod550.data, da.Array)

    # Check consistency
    xr.testing.assert_allclose(res_eager, res_lazy.compute())

    # Check history
    assert "history" in res_eager.attrs
    assert "Read ICAP-MME data." in res_eager.attrs["history"]


if __name__ == "__main__":
    pytest.main([__file__])
