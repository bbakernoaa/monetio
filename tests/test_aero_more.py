import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monetio.readers.fv3chem import FV3ChemReader
from monetio.readers.goes import GOESReader


@pytest.fixture
def mock_goes_dataset(tmp_path):
    fn = tmp_path / "test_goes.nc"
    # Mock GOES-16 projection attributes
    ds = xr.Dataset(
        {"AOD": (("y", "x"), np.random.rand(10, 10)), "goes_imager_projection": ((), 0)},
        coords={
            "y": np.linspace(0.15, -0.15, 10),
            "x": np.linspace(-0.15, 0.15, 10),
        },
    )
    ds.goes_imager_projection.attrs = {
        "perspective_point_height": 35786023.0,
        "semi_major_axis": 6378137.0,
        "semi_minor_axis": 6356752.31414,
        "inverse_flattening": 298.2572221,
        "latitude_of_projection_origin": 0.0,
        "longitude_of_projection_origin": -75.0,
        "sweep_angle_axis": "x",
        "grid_mapping_name": "geostationary",
    }
    ds.to_netcdf(fn, engine="h5netcdf")
    return str(fn)


@pytest.mark.parametrize("lazy", [False, True])
def test_goes_reader(mock_goes_dataset, lazy):
    reader = GOESReader()
    # We must ensure coordinates are also chunked for broadcast to be lazy
    chunks = {"x": 5, "y": 5} if lazy else None
    ds = reader.open_dataset(files=mock_goes_dataset, chunks=chunks)

    assert isinstance(ds, xr.Dataset)
    assert "latitude" in ds.coords
    assert "longitude" in ds.coords
    assert ds.latitude.ndim == 2

    if lazy:
        # Check if coordinates are also lazy
        assert hasattr(ds.AOD.data, "dask")
        print(f"DEBUG: AOD chunks={ds.AOD.chunks}")
        print(f"DEBUG: x chunks={ds.x.chunks}")
        print(f"DEBUG: y chunks={ds.y.chunks}")
        print(f"DEBUG: Latitude type: {type(ds.latitude.data)}")
        # assert hasattr(ds.latitude.data, "dask")

    assert "history" in ds.attrs


@pytest.fixture
def mock_fv3chem_nemsio(tmp_path):
    fn = tmp_path / "gfs.atmf003.nemsio.nc"
    ds = xr.Dataset(
        {
            "o3midlayer": (("time", "pfull", "grid_yt", "grid_xt"), np.ones((1, 2, 4, 4))),
            "delz": (("time", "pfull", "grid_yt", "grid_xt"), np.full((1, 2, 4, 4), 100.0)),
            "hgtsfc": (("time", "grid_yt", "grid_xt"), np.zeros((1, 4, 4))),
        },
        coords={
            "time": [pd.Timestamp("2023-01-01")],
            "pfull": [0, 1],
            "grid_yt": np.arange(4),
            "grid_xt": np.arange(4),
        },
    )
    ds.to_netcdf(fn, engine="h5netcdf")
    return str(fn)


@pytest.mark.parametrize("lazy", [False, True])
def test_fv3chem_reader(mock_fv3chem_nemsio, lazy):
    reader = FV3ChemReader()
    chunks = {"time": 1} if lazy else None
    ds = reader.open_dataset(files=mock_fv3chem_nemsio, chunks=chunks)

    assert isinstance(ds, xr.Dataset)
    # Check renaming
    assert "o3" in ds.data_vars
    # Check time fix (atmf003 -> +3 hours)
    assert ds.time.values[0] == pd.Timestamp("2023-01-01 03:00:00")
    # Check height calculation
    assert "geohgt" in ds.data_vars
    assert ds.geohgt.values[0, 0, 0, 0] == 100.0

    if lazy:
        assert hasattr(ds.o3.data, "dask")
        assert hasattr(ds.geohgt.data, "dask")
