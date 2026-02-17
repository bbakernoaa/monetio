import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monetio.readers.ufs import UFSReader


@pytest.fixture
def mock_ufs_dataset(tmp_path):
    fn = tmp_path / "test_ufs.nc"

    # Create a small mock UFS dataset
    nt, nz, ny, nx = 2, 3, 4, 5
    ds = xr.Dataset(
        {
            "o3": (("time", "pfull", "grid_yt", "grid_xt"), np.random.rand(nt, nz, ny, nx)),
            "tmp": (("time", "pfull", "grid_yt", "grid_xt"), np.full((nt, nz, ny, nx), 290.0)),
            "pressfc": (("time", "grid_yt", "grid_xt"), np.full((nt, ny, nx), 101325.0)),
            "ak": (("phalf",), [0, 100, 200, 300]),
            "bk": (("phalf",), [0, 0.1, 0.2, 0.3]),
            "lat": (("grid_yt", "grid_xt"), np.zeros((ny, nx))),
            "lon": (("grid_yt", "grid_xt"), np.zeros((ny, nx))),
        },
        coords={
            "time": pd.date_range("2023-01-01", periods=nt, freq="h"),
            "pfull": np.arange(nz),
            "phalf": np.arange(nz + 1),
            "grid_yt": np.arange(ny),
            "grid_xt": np.arange(nx),
        },
    )
    ds["o3"].attrs["units"] = "ppmv"
    ds["tmp"].attrs["units"] = "K"
    ds.to_netcdf(fn, engine="h5netcdf")
    return str(fn)


@pytest.mark.parametrize("lazy", [False, True])
def test_ufs_reader(mock_ufs_dataset, lazy):
    reader = UFSReader()
    chunks = {"time": 1} if lazy else None
    ds = reader.open_dataset(files=mock_ufs_dataset, chunks=chunks)

    assert isinstance(ds, xr.Dataset)
    assert "o3" in ds.data_vars
    assert ds.o3.attrs["units"] == "ppbv"  # Converted from ppmv

    if lazy:
        assert ds.o3.chunks is not None

    # Check if pressure was calculated
    assert "pres_pa_mid" in ds.data_vars

    # Check history
    assert "history" in ds.attrs
    assert "Read UFS-AQM data" in ds.attrs["history"]


def test_ufs_lazy_pm25(tmp_path):
    fn = tmp_path / "test_ufs_pm.nc"
    ds = xr.Dataset(
        {
            "aso4j": (("time", "pfull", "grid_yt", "grid_xt"), np.ones((1, 1, 1, 1))),
            "asoil": (("time", "pfull", "grid_yt", "grid_xt"), np.ones((1, 1, 1, 1))),
        },
        coords={
            "time": [pd.Timestamp("2023-01-01")],
            "pfull": [0],
            "grid_yt": [0],
            "grid_xt": [0],
        },
    )
    ds.to_netcdf(fn, engine="h5netcdf")

    reader = UFSReader()
    # PM25 calculation should be triggered
    ds_out = reader.open_dataset(files=str(fn))
    assert "PM25" in ds_out.data_vars
    # aso4j (accumulation) weight 1.0 + asoil (coarse) weight 0.2 = 1.2
    assert ds_out.PM25.values[0, 0, 0, 0] == pytest.approx(1.2)
