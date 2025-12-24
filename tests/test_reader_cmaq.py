import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monetio.readers.cmaq import CMAQReader


@pytest.fixture
def fake_cmaq_file(tmp_path):
    """Create a fake CMAQ file for testing."""
    filepath = tmp_path / "cmaq_test.nc"
    ds = xr.Dataset(
        {
            "TFLAG": (
                ("TSTEP", "VAR", "DATE-TIME"),
                np.array([[[2023001, 20000]]], dtype=np.int32),
            ),
            "O3": (
                ("TSTEP", "LAY", "ROW", "COL"),
                np.random.rand(1, 1, 10, 10).astype(np.float32),
                {"units": "ppmV"},
            ),
            "NO2": (
                ("TSTEP", "LAY", "ROW", "COL"),
                np.random.rand(1, 1, 10, 10).astype(np.float32),
                {"units": "ppmV"},
            ),
            "PM25_TOT": (
                ("TSTEP", "LAY", "ROW", "COL"),
                np.random.rand(1, 1, 10, 10).astype(np.float32),
                {"units": r"$\mu g m^{-3}$"},
            ),
        },
        coords={
            "TSTEP": (("TSTEP",), [0]),
            "LAY": (("LAY",), [0]),
            "ROW": (("ROW",), np.arange(10)),
            "COL": (("COL",), np.arange(10)),
        },
        attrs={
            "GRID_NAME": "TEST_GRID",
            "XORIG": -100.0,
            "YORIG": 20.0,
            "XCELL": 1.0,
            "YCELL": 1.0,
            "NCOLS": 10,
            "NROWS": 10,
            "NLAYS": 1,
            "GDTYP": 2,
            "P_ALP": 33.0,
            "P_BET": 45.0,
            "P_GAM": -97.0,
            "XCENT": -97.0,
            "YCENT": 40.0,
        },
    )
    ds.to_netcdf(filepath)
    return filepath


def test_cmaq_reader_opens_and_corrects(fake_cmaq_file):
    """Test that the CMAQReader opens a file, applies corrections, and adds history."""
    reader = CMAQReader()
    ds = reader.open_dataset(str(fake_cmaq_file))

    # 1. Test Time Coordinate
    assert "time" in ds.coords
    expected_time = pd.to_datetime("2023-01-01 02:00:00")
    assert ds.time.values[0] == expected_time

    # 2. Test Coordinate and Dimension Renaming
    assert "x" in ds.dims
    assert "y" in ds.dims
    assert "z" in ds.dims
    assert "latitude" in ds.coords
    assert "longitude" in ds.coords

    # 3. Test Unit Conversion
    assert ds["O3"].attrs["units"] == "ppbV"
    assert ds["NO2"].attrs["units"] == "ppbV"
    # Test that original values were multiplied by 1000
    # Note: This is an indirect check since we can't easily access the original data
    # without re-reading it. For this test, we assume the initial random values are < 1.
    assert ds["O3"].mean() > 1
    assert ds["NO2"].mean() > 1

    # 4. Test Lazy Variable Addition
    assert "PM25" in ds.variables

    # 5. Test History Attribute
    assert "history" in ds.attrs
    assert "Applied MONET-standard corrections" in ds.attrs["history"]
    assert "Converted ppmV to ppbV: True" in ds.attrs["history"]
