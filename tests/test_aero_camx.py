import numpy as np
import pytest
import xarray as xr

from monetio.readers.camx import CAMxReader


def mock_camx_dataset():
    """Create a mock CAMx dataset."""
    # Dimensions
    ntime = 2
    nlay = 1
    nrow = 3
    ncol = 4

    # Time flags (YYYYDDD, HHMMSS)
    # 2023-01-01 00:00:00 and 01:00:00
    tflag_data = [
        [[2023001, 0], [2023001, 10000]],  # For Var 1
        [[2023001, 0], [2023001, 10000]],  # For Var 2
    ]
    # In some CAMx formats it is (TSTEP, VAR, DATE_TIME)
    tflag = xr.DataArray(
        np.array(tflag_data).transpose(1, 0, 2), dims=("TSTEP", "VAR", "DATE_TIME"), name="TFLAG"
    )

    # Coordinates and data variables
    ds = xr.Dataset(
        {
            "O3": (("TSTEP", "LAY", "ROW", "COL"), np.random.rand(ntime, nlay, nrow, ncol)),
            "NO": (("TSTEP", "LAY", "ROW", "COL"), np.random.rand(ntime, nlay, nrow, ncol)),
            "TFLAG": tflag,
        }
    )

    # Grid attributes (LCC example)
    ds.attrs["GDTYP"] = 2
    ds.attrs["P_ALP"] = 30.0
    ds.attrs["P_BET"] = 60.0
    ds.attrs["P_GAM"] = -100.0
    ds.attrs["XCENT"] = -100.0
    ds.attrs["YCENT"] = 40.0
    ds.attrs["XORIG"] = -1000.0
    ds.attrs["YORIG"] = -1000.0
    ds.attrs["XCELL"] = 500.0
    ds.attrs["YCELL"] = 500.0
    ds.attrs["NCOLS"] = ncol
    ds.attrs["NROWS"] = nrow
    ds.attrs["NLAYS"] = nlay

    # Add units
    ds.O3.attrs["units"] = "ppm"
    ds.NO.attrs["units"] = "ppm"

    return ds


def test_camx_eager_lazy(tmp_path):
    ds_mock = mock_camx_dataset()
    fname = tmp_path / "test_camx.nc"
    ds_mock.to_netcdf(fname, engine="h5netcdf")

    reader = CAMxReader()

    # Eager Mode
    ds_eager = reader.open_dataset(files=str(fname), lazy=False, engine="h5netcdf")
    assert isinstance(ds_eager, xr.Dataset)
    assert "time" in ds_eager.coords
    assert "latitude" in ds_eager.coords
    assert "longitude" in ds_eager.coords
    assert "O3" in ds_eager.data_vars
    assert ds_eager.O3.attrs["units"] == "ppbV"  # Converted from ppm

    # Lazy Mode
    # Note: Using chunks={} in open_dataset kwargs triggers dask
    ds_lazy = reader.open_dataset(files=str(fname), chunks={"time": 1}, engine="h5netcdf")
    assert isinstance(ds_lazy, xr.Dataset)
    assert hasattr(ds_lazy.O3.data, "dask")

    # Verify values
    xr.testing.assert_allclose(
        ds_eager.drop_vars("history", errors="ignore"),
        ds_lazy.compute().drop_vars("history", errors="ignore"),
    )

    # Check diagnostics
    assert "NOx" in ds_eager.data_vars
    # NOX = NO + NO2. In our mock we only have NO, so NOx should be NO.
    # Actually add_lazy_diagnostic only adds if variables exist.
    # In camx_specs.py, NOx is [NO, NOX]. If NOX is missing, it still adds NO?
    # No, it checks available_vars. If only NO is available, NOx = NO.


if __name__ == "__main__":
    pytest.main([__file__])
