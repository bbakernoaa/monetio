import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monetio.readers.tropomi import tropomi_preprocess


@pytest.mark.parametrize("lazy", [True, False])
def test_tropomi_preprocess(lazy):
    # Create dummy TROPOMI dataset
    # Dimensions: scanline, ground_pixel
    # Coordinates: latitude, longitude, time
    # Data: delta_time, nitrogendioxide_tropospheric_column

    ref_time = pd.Timestamp("2023-01-01T00:00:00")
    delta_times = np.array([0, 1000, 2000], dtype="int32")  # ms

    ds = xr.Dataset(
        {
            "nitrogendioxide_tropospheric_column": (
                ("scanline", "ground_pixel"),
                np.random.rand(3, 4).astype(np.float32),
            ),
            "delta_time": (("scanline",), delta_times),
        },
        coords={
            "latitude": (("scanline", "ground_pixel"), np.random.rand(3, 4).astype(np.float32)),
            "longitude": (("scanline", "ground_pixel"), np.random.rand(3, 4).astype(np.float32)),
            "time": ((), np.datetime64(ref_time)),
        },
    )

    if lazy:
        ds = ds.chunk({"scanline": 2, "ground_pixel": 2})

    ds_out = tropomi_preprocess(ds)

    assert "latitude" in ds_out.coords
    assert "longitude" in ds_out.coords
    assert "time" in ds_out.dims
    assert ds_out.time.size == 3
    assert ds_out.nitrogendioxide_tropospheric_column.dims == ("time", "x")

    # Check time values
    expected_times = [ref_time + pd.Timedelta(milliseconds=ms) for ms in delta_times]
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(ds_out.time.values), pd.DatetimeIndex(expected_times)
    )

    if lazy:
        assert ds_out.nitrogendioxide_tropospheric_column.chunks is not None
