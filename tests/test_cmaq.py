import numpy as np
import pytest
import xarray as xr
import pandas as pd
from monetio.readers.cmaq import CMAQReader

def create_mock_cmaq_dataset(n_times=2, n_lay=1, n_rows=10, n_cols=10, use_dask=False):
    """Creates a mock CMAQ dataset."""
    times = pd.date_range("2023-01-01", periods=n_times, freq="h")

    # Create TFLAG: (time, nvars, 2)
    # TFLAG format: [YYYYDDD, HHMMSS]
    tflag = np.zeros((n_times, 1, 2), dtype=np.int32)
    for i, t in enumerate(times):
        tflag[i, 0, 0] = int(t.strftime("%Y%j"))
        tflag[i, 0, 1] = int(t.strftime("%H%M%S"))

    ds = xr.Dataset(
        coords={
            "TSTEP": np.arange(n_times),
            "LAY": np.arange(n_lay),
            "ROW": np.arange(n_rows),
            "COL": np.arange(n_cols),
            "VAR": [b"O3"],
            "DATE_TIME": np.arange(2),
        },
        data_vars={
            "TFLAG": (("TSTEP", "VAR", "DATE_TIME"), tflag),
            "O3": (("TSTEP", "LAY", "ROW", "COL"), np.random.rand(n_times, n_lay, n_rows, n_cols).astype(np.float32)),
            "ASO4J": (("TSTEP", "LAY", "ROW", "COL"), np.random.rand(n_times, n_lay, n_rows, n_cols).astype(np.float32)),
            "ANO3J": (("TSTEP", "LAY", "ROW", "COL"), np.random.rand(n_times, n_lay, n_rows, n_cols).astype(np.float32)),
            "ANH4J": (("TSTEP", "LAY", "ROW", "COL"), np.random.rand(n_times, n_lay, n_rows, n_cols).astype(np.float32)),
        },
    )

    # Set IOAPI attributes
    ds.attrs.update({
        "IOAPI_VERSION": "mock",
        "GDTYP": 2,  # Lambert
        "P_ALP": 33.0,
        "P_BET": 45.0,
        "YCENT": 40.0,
        "P_GAM": -97.0,
        "XCENT": -97.0,
        "XORIG": -100000.0,
        "YORIG": -100000.0,
        "XCELL": 12000.0,
        "YCELL": 12000.0,
        "NCOLS": n_cols,
        "NROWS": n_rows,
    })

    ds.O3.attrs["units"] = "ppmV"
    ds.ASO4J.attrs["units"] = "micrograms/m**3"
    ds.ANO3J.attrs["units"] = "micrograms/m**3"
    ds.ANH4J.attrs["units"] = "micrograms/m**3"

    if use_dask:
        ds = ds.chunk({"TSTEP": 1})

    return ds

def test_cmaq_reader_eager_vs_lazy(tmp_path):
    """Verifies CMAQ reader with both Eager and Lazy data."""
    # 1. Setup mock files
    ds_eager = create_mock_cmaq_dataset(use_dask=False)
    file_path = str(tmp_path / "mock_cmaq.nc")
    ds_eager.to_netcdf(file_path, engine="h5netcdf")

    reader = CMAQReader()

    # 2. Test Eager
    ds_out_eager = reader.open_dataset(file_path, use_dask=False, engine="h5netcdf")
    assert isinstance(ds_out_eager.O3.data, np.ndarray)
    assert ds_out_eager.O3.attrs["units"] == "ppbV"
    assert "PM25" in ds_out_eager.data_vars
    assert "Read CMAQ data" in ds_out_eager.attrs["history"]

    # 3. Test Lazy
    ds_out_lazy = reader.open_dataset(file_path, use_dask=True, engine="h5netcdf")
    # Check if it is a dask array
    assert hasattr(ds_out_lazy.O3.data, "dask")

    # 4. Verify results are identical
    xr.testing.assert_allclose(ds_out_eager.compute(), ds_out_lazy.compute())

def test_cmaq_coordinates(tmp_path):
    """Verifies that coordinates are correctly generated."""
    ds = create_mock_cmaq_dataset()
    file_path = str(tmp_path / "coord_test.nc")
    ds.to_netcdf(file_path, engine="h5netcdf")

    reader = CMAQReader()
    ds_out = reader.open_dataset(file_path, engine="h5netcdf")

    # Check dimensions (they are renamed to x, y, z in open_dataset)
    assert "x" in ds_out.dims
    assert "y" in ds_out.dims
    assert ds_out.longitude.dims == ("y", "x")

    # Latitude should increase with row index (y) in our mock dataset (Northern Hemisphere)
    assert ds_out.latitude.isel(y=-1, x=0) > ds_out.latitude.isel(y=0, x=0)

def test_add_lazy_diagnostic():
    """Test the generic diagnostic addition logic."""
    from monetio.readers.cmaq import add_lazy_diagnostic
    from monetio.readers.cmaq_specs import DiagnosticSpec

    ds = xr.Dataset(
        data_vars={
            "A": (("x",), [1.0, 2.0]),
            "B": (("x",), [3.0, 4.0]),
        }
    )

    spec = DiagnosticSpec(
        variables=["A", "B", "C"],
        weights=[1.0, 0.5, 2.0],
        units="test_units",
        long_name="test_long",
        name="test_name"
    )

    # C is missing, should still work with A and B
    ds_out = add_lazy_diagnostic(ds, "TEST", spec)

    expected = ds["A"] * 1.0 + ds["B"] * 0.5
    xr.testing.assert_allclose(ds_out["TEST"], expected)
    assert ds_out["TEST"].attrs["units"] == "test_units"

def test_get_times():
    """Test time extraction logic."""
    from monetio.readers.cmaq import _get_times
    import numpy as np

    # TFLAG: YYYYDDD, HHMMSS
    tflag = np.array([
        [2023001, 0],
        [2023001, 10000]
    ], dtype=np.int32).reshape(2, 1, 2)

    ds = xr.Dataset(
        data_vars={
            "TFLAG": (("TSTEP", "VAR", "DATE-TIME"), tflag)
        }
    )

    ds_out = _get_times(ds, drop_duplicates=False)
    assert "time" in ds_out.coords
    assert ds_out.time[0].values == np.datetime64("2023-01-01T00:00:00")
    assert ds_out.time[1].values == np.datetime64("2023-01-01T01:00:00")
