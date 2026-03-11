import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monetio.readers.geoms import geoms_preprocess


def create_mock_geoms_ds(lazy=False):
    """Creates a mock GEOMS dataset."""
    nt = 5
    nz = 10

    # Mock MJD2000 times (approx 2023-01-01)
    times = np.linspace(8401, 8402, nt)

    # Generate fixed random data
    o3_data = np.arange(nt * nz).reshape(nt, nz).astype(float)

    data = {
        "DATETIME": (("fakeDim0datetime",), times),
        "ALTITUDE": (("fakeDim0altitude",), np.linspace(0, 10000, nz)),
        "O3.MIXING.RATIO.VOLUME": (("fakeDim0datetime", "fakeDim0altitude"), o3_data),
        "LATITUDE.INSTRUMENT": (("fakeDim0latitude",), [40.0]),
        "LONGITUDE.INSTRUMENT": (("fakeDim0longitude",), [-105.0]),
        "ALTITUDE.INSTRUMENT": (("fakeDim0altinst",), [1600.0]),
        "STRING_VAR": (("fakeDim0datetime",), np.array([b"test"] * nt, dtype=object)),
    }

    ds = xr.Dataset(data_vars=data, attrs={"DATA_SOURCE": "MOCK"})

    if lazy:
        ds = ds.chunk({"fakeDim0datetime": 2, "fakeDim0altitude": -1})

    return ds


def test_geoms_protocol_compliance():
    """Verify GEOMS processing is backend-agnostic and lazy-friendly."""
    ds_base = create_mock_geoms_ds(lazy=False)
    ds_eager = ds_base.copy(deep=True)
    ds_lazy = ds_base.chunk({"fakeDim0datetime": 2, "fakeDim0altitude": -1})

    # Test Preprocess Eager
    res_eager = geoms_preprocess(ds_eager)

    # Test Preprocess Lazy
    res_lazy = geoms_preprocess(ds_lazy)

    # Check laziness
    assert isinstance(res_lazy.o3_mixing_ratio_volume.data, da.Array)
    assert isinstance(res_lazy.time.data, da.Array)

    # Check consistency
    xr.testing.assert_allclose(res_eager, res_lazy.compute())

    # Check time conversion
    expected_time = pd.to_datetime(ds_base.DATETIME.values + 2451544.5, unit="D", origin="julian")
    # Use xr.testing for datetime comparisons, but drop coords for direct comparison if needed
    xr.testing.assert_allclose(
        res_eager.time.drop_vars(res_eager.time.coords),
        xr.DataArray(expected_time.values.astype("datetime64[ns]"), dims="time"),
    )

    # Check string decoding
    assert res_eager.string_var.values[0] == "test"

    # Check dimension names
    assert "time" in res_eager.dims
    assert "altitude" in res_eager.dims
    assert "latitude" in res_eager.coords
    assert "longitude" in res_eager.coords


if __name__ == "__main__":
    pytest.main([__file__])
