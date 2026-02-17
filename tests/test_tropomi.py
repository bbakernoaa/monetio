import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monetio.readers.tropomi import TROPOMIReader, tropomi_preprocess


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

    # Check time values with robust precision handling
    expected_times = [ref_time + pd.Timedelta(milliseconds=ms) for ms in delta_times]
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(ds_out.time.values).astype("datetime64[ns]"),
        pd.DatetimeIndex(expected_times).astype("datetime64[ns]"),
    )

    if lazy:
        assert ds_out.nitrogendioxide_tropospheric_column.chunks is not None


@pytest.mark.parametrize("lazy", [True, False])
def test_tropomi_enhanced_features(lazy):
    # Test pressure and quality flagging
    ds = xr.Dataset(
        {
            "no2": (("scanline", "ground_pixel"), np.ones((3, 4), dtype=np.float32)),
            "qa_value": (
                ("scanline", "ground_pixel"),
                np.array([[0, 0.5, 0.8, 1]] * 3, dtype=np.float32),
            ),
            "surface_pressure": (
                ("scanline", "ground_pixel"),
                np.full((3, 4), 101325.0, dtype=np.float32),
            ),
            "tm5_constant_a": (
                ("layer", "vertices"),
                np.array([[0, 100], [100, 200]], dtype=np.float32),
            ),
            "tm5_constant_b": (
                ("layer", "vertices"),
                np.array([[0, 0.1], [0.1, 0.2]], dtype=np.float32),
            ),
            "tm5_tropopause_layer_index": (
                ("scanline", "ground_pixel"),
                np.zeros((3, 4), dtype=int),
            ),
            "delta_time": (("scanline",), np.array([0, 1000, 2000], dtype="int32")),
        },
        coords={
            "latitude": (("scanline", "ground_pixel"), np.zeros((3, 4))),
            "longitude": (("scanline", "ground_pixel"), np.zeros((3, 4))),
            "time": ((), np.datetime64("2023-01-01")),
        },
    )

    if lazy:
        ds = ds.chunk({"scanline": 2, "layer": 1})

    ds_out = tropomi_preprocess(ds, calculate_pressure=True, qa_threshold=0.75)

    # 1. Check Quality Flagging
    # no2 should be NaN where qa_value < 0.75
    # qa_value [0, 0.5, 0.8, 1] -> [NaN, NaN, 1.0, 1.0]
    no2_vals = ds_out.no2.values
    assert np.isnan(no2_vals[0, 0])
    assert np.isnan(no2_vals[0, 1])
    assert no2_vals[0, 2] == 1.0

    # 2. Check Pressure
    assert "pres_pa_mid" in ds_out.data_vars
    assert "z" in ds_out.pres_pa_mid.dims
    assert "time" in ds_out.pres_pa_mid.dims
    assert "x" in ds_out.pres_pa_mid.dims

    # 3. Check Tropopause Pressure
    assert "troppres" in ds_out.data_vars
    assert ds_out.troppres.dims == ("time", "x")

    if lazy:
        assert ds_out.pres_pa_mid.chunks is not None


def test_tropomi_multi_group(tmp_path, monkeypatch):
    # Simulate multi-group file opening using mocked XarrayDriver
    # PRODUCT group
    ds_prod = xr.Dataset(
        {"no2": (("scanline", "ground_pixel"), np.ones((3, 4), dtype=np.float32))},
        coords={
            "latitude": (("scanline", "ground_pixel"), np.zeros((3, 4))),
            "longitude": (("scanline", "ground_pixel"), np.zeros((3, 4))),
            "time": ((), np.datetime64("2023-01-01")),
        },
    )
    # INPUT_DATA group
    ds_input = xr.Dataset(
        {
            "surface_pressure": (
                ("scanline", "ground_pixel"),
                np.full((3, 4), 101325.0, dtype=np.float32),
            )
        }
    )

    class MockDriver:
        def open(self, files, **kwargs):
            group = kwargs.get("group")
            if group == "PRODUCT":
                return ds_prod
            if group == "PRODUCT/SUPPORT_DATA/INPUT_DATA":
                return ds_input
            return xr.Dataset()

    # We need to monkeypatch the driver instance in TROPOMIReader
    reader = TROPOMIReader()
    monkeypatch.setattr(reader, "driver", MockDriver())

    # Open both groups
    ds_merged = reader.open_dataset(
        ["fake_file.nc"],
        group=["PRODUCT", "PRODUCT/SUPPORT_DATA/INPUT_DATA"],
        calculate_pressure=False,
    )

    assert "no2" in ds_merged.data_vars
    assert "surface_pressure" in ds_merged.data_vars
    assert ds_merged.no2.shape == (3, 4)
    assert ds_merged.surface_pressure.shape == (3, 4)
    assert "time" in ds_merged.coords
