from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from monetio.readers.cmaq import CMAQReader
from monetio.readers.drivers import XarrayDriver


def test_cmaq_reader_eager_lazy():
    """Verify CMAQReader logic on both Eager (NumPy) and Lazy (Dask) data."""
    # 1. Create a mock IOAPI dataset
    tflag_data = np.zeros((2, 1, 2), dtype=np.int32)
    tflag_data[0, 0, 0] = 2023001  # 2023-01-01
    tflag_data[0, 0, 1] = 120000  # 12:00:00
    tflag_data[1, 0, 0] = 2023001
    tflag_data[1, 0, 1] = 130000  # 13:00:00

    ds = xr.Dataset(
        {
            "TFLAG": (("TSTEP", "VAR", "DATE-TIME"), tflag_data),
            "O3": (("TSTEP", "LAY", "ROW", "COL"), np.ones((2, 1, 5, 5))),
            "NO": (("TSTEP", "LAY", "ROW", "COL"), np.ones((2, 1, 5, 5))),
            "NO2": (("TSTEP", "LAY", "ROW", "COL"), np.ones((2, 1, 5, 5))),
        }
    )
    # IOAPI attributes for _get_latlon
    ds.attrs["P_ALP"] = 30.0
    ds.attrs["P_BET"] = 60.0
    ds.attrs["P_GAM"] = -97.0
    ds.attrs["XCENT"] = -97.0
    ds.attrs["YCENT"] = 40.0
    ds.attrs["XORIG"] = -100000.0
    ds.attrs["YORIG"] = -100000.0
    ds.attrs["XCELL"] = 4000.0
    ds.attrs["YCELL"] = 4000.0
    ds.attrs["NCOLS"] = 5
    ds.attrs["NROWS"] = 5
    ds.attrs["GDTYP"] = 2  # Lambert
    ds.attrs["IOAPI_VERSION"] = "3.2"

    ds.O3.attrs["units"] = "ppmV"
    ds.NO.attrs["units"] = "ppmV"
    ds.NO2.attrs["units"] = "ppmV"

    reader = CMAQReader()

    # 2. Test Eager (NumPy)
    ds_eager = reader._prepare_ds(
        ds.copy(), earth_radius=6370000, convert_to_ppb=True, drop_duplicates=False
    )

    assert "time" in ds_eager.coords
    assert "latitude" in ds_eager.coords
    assert "longitude" in ds_eager.coords
    assert ds_eager.O3.attrs["units"] == "ppbV"
    assert (ds_eager.O3 == 1000.0).all()
    assert "NOx" in ds_eager.data_vars
    assert ds_eager.NOx.attrs["units"] == "ppbV"
    assert (ds_eager.NOx == 2000.0).all()
    assert ds_eager.time.dtype == "datetime64[ns]"

    # 3. Test Lazy (Dask)
    try:
        import dask.array as da  # noqa: F401

        ds_lazy = ds.copy().chunk({"TSTEP": 1})
        ds_lazy_processed = reader._prepare_ds(
            ds_lazy, earth_radius=6370000, convert_to_ppb=True, drop_duplicates=False
        )

        assert "time" in ds_lazy_processed.coords
        assert hasattr(ds_lazy_processed.O3.data, "dask")

        # Verify values match
        xr.testing.assert_allclose(ds_eager, ds_lazy_processed.compute())
    except ImportError:
        pytest.skip("Dask not installed")


def test_xarray_driver_wildcard():
    """Test XarrayDriver wildcard handling."""
    with patch("xarray.open_mfdataset") as mock_mf:
        XarrayDriver.open_dataset("path/to/*.nc")
        mock_mf.assert_called_once_with("path/to/*.nc")

    with patch("xarray.open_dataset") as mock_open:
        XarrayDriver.open_dataset("path/to/single.nc")
        mock_open.assert_called_once_with("path/to/single.nc")


def test_time_parsing_direct():
    """Test parse_ioapi_times directly."""
    from monetio.readers.time_utils import parse_ioapi_times

    tflag_data = np.zeros((1, 1, 2), dtype=np.int32)
    tflag_data[0, 0, 0] = 2023032  # 2023 Feb 01
    tflag_data[0, 0, 1] = 153045

    tflag = xr.DataArray(tflag_data, dims=("TSTEP", "VAR", "DATE-TIME"))
    times = parse_ioapi_times(tflag)

    expected = np.datetime64("2023-02-01T15:30:45")
    assert times.values[0] == expected
