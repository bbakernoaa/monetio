import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monetio.readers.omps_nadir import OMPSNadirReader, omps_nadir_preprocess


@pytest.mark.parametrize("lazy", [True, False])
def test_omps_v8toz_preprocess(lazy):
    # Create dummy V8TOZ dataset
    # nTimes, nIFOV
    n_times = 5
    n_ifov = 10

    # Microseconds since 1958-01-01
    origin = pd.Timestamp("1958-01-01")
    target_time = pd.Timestamp("2024-01-01 18:45:10")
    time_val = (target_time - origin).total_seconds() * 1e6

    scan_times = np.full(n_times, time_val) + np.arange(n_times) * 1e6  # add 1s each

    ds = xr.Dataset(
        {
            "ColumnAmountO3": (
                ("nTimes", "nIFOV"),
                np.random.rand(n_times, n_ifov).astype(np.float32),
            ),
            "ScanTime": (("nTimes",), scan_times),
            "Latitude": (("nTimes", "nIFOV"), np.random.rand(n_times, n_ifov).astype(np.float32)),
            "Longitude": (("nTimes", "nIFOV"), np.random.rand(n_times, n_ifov).astype(np.float32)),
        }
    )

    if lazy:
        ds = ds.chunk({"nTimes": 2})

    ds_out = omps_nadir_preprocess(ds, product="v8toz")

    assert "latitude" in ds_out.coords
    assert "longitude" in ds_out.coords
    assert "time" in ds_out.coords
    assert ds_out.y.size == n_times
    assert ds_out.x.size == n_ifov

    # Check time
    assert ds_out.time.values[0] == np.datetime64(target_time)

    if lazy:
        assert ds_out.ozone_column.chunks is not None


def test_omps_nadir_reader_inference():
    reader = OMPSNadirReader()
    # Basic check that reader can be instantiated
    assert isinstance(reader, OMPSNadirReader)


@pytest.mark.parametrize("product", ["nmto3_l2", "nmto3_l3"])
def test_omps_nadir_nasa_fallback(product):
    # Create dummy NASA style dataset
    ds = xr.Dataset(
        {
            "ColumnAmountO3": (("scanline", "ground_pixel"), np.ones((3, 4), dtype=np.float32)),
            "Time": (("scanline",), np.zeros(3)),
        },
        coords={
            "Latitude": (("scanline", "ground_pixel"), np.zeros((3, 4))),
            "Longitude": (("scanline", "ground_pixel"), np.zeros((3, 4))),
        },
    )

    # For NASA products, it should still work
    ds_out = omps_nadir_preprocess(ds, product=product)

    assert "latitude" in ds_out.coords
    assert "longitude" in ds_out.coords
    assert "ozone_column" in ds_out.data_vars
    assert "time" in ds_out.coords or "time" in ds_out.dims
