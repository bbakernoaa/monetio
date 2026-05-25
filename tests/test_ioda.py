import os

import netCDF4
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import monetio
from monetio.readers.ioda import export_to_ioda


@pytest.fixture
def mock_ioda_file(tmp_path):
    fn = tmp_path / "test_ioda.nc"

    with netCDF4.Dataset(fn, "w", format="NETCDF4") as nc:
        # Create groups
        meta = nc.createGroup("MetaData")
        obs = nc.createGroup("ObsValue")
        hofx = nc.createGroup("HofX")

        # Core dimension
        nlocs = 5
        meta.createDimension("nlocs", nlocs)
        obs.createDimension("nlocs", nlocs)
        hofx.createDimension("nlocs", nlocs)

        # MetaData variables
        lat = meta.createVariable("latitude", "f4", ("nlocs",))
        lat[:] = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        lon = meta.createVariable("longitude", "f4", ("nlocs",))
        lon[:] = np.array([-100.0, -90.0, -80.0, -70.0, -60.0])

        dt = meta.createVariable("dateTime", str, ("nlocs",))
        dt[0] = "2023-01-01T00:00:00Z"
        dt[1] = "2023-01-01T01:00:00Z"
        dt[2] = "2023-01-01T02:00:00Z"
        dt[3] = "2023-01-01T03:00:00Z"
        dt[4] = "2023-01-01T04:00:00Z"

        # ObsValue variables
        o3 = obs.createVariable("ozone_conc", "f4", ("nlocs",))
        o3[:] = np.array([30.0, 35.0, 40.0, 45.0, 50.0])

        # HofX variables
        o3_sim = hofx.createVariable("ozone_conc", "f4", ("nlocs",))
        o3_sim[:] = np.array([31.0, 36.0, 41.0, 46.0, 51.0])

    return str(fn)


def test_ioda_read(mock_ioda_file):
    ds = monetio.load("ioda", files=mock_ioda_file)

    assert isinstance(ds, xr.Dataset)
    assert "node" in ds.dims
    assert ds.sizes["node"] == 5
    assert "mesh" in ds.variables
    assert ds.mesh.attrs["cf_role"] == "mesh_topology"

    # Check variables
    assert "latitude" in ds.variables
    assert "longitude" in ds.variables
    assert "time" in ds.variables
    assert "ozone_conc" in ds.variables
    assert "ozone_conc_sim" in ds.variables

    # Check data values
    np.testing.assert_allclose(ds.latitude.values, [10, 20, 30, 40, 50])
    np.testing.assert_allclose(ds.ozone_conc.values, [30, 35, 40, 45, 50])
    np.testing.assert_allclose(ds.ozone_conc_sim.values, [31, 36, 41, 46, 51])

    # Check time conversion
    assert np.issubdtype(ds.time.dtype, np.datetime64)
    assert ds.time.values[0] == pd.Timestamp("2023-01-01T00:00:00")


def test_ioda_export(tmp_path):
    # Create a dummy monet-like dataset (multi-dimensional)
    times = pd.date_range("2023-01-01", periods=2, freq="h")
    sites = ["site1", "site2"]

    ds = xr.Dataset(
        data_vars={
            "ozone": (("time", "site"), [[30.0, 31.0], [32.0, 33.0]]),
            "error": (("time", "site"), [[1.0, 1.1], [1.2, 1.3]]),
        },
        coords={
            "time": times,
            "site": sites,
            "latitude": (("site",), [34.0, 35.0]),
            "longitude": (("site",), [-118.0, -117.0]),
        },
    )

    output_path = str(tmp_path / "exported_ioda.nc")
    mapping = {
        "ozone": ("ObsValue", "ozone_conc"),
        "error": ("ObsError", "ozone_conc"),
        "latitude": ("MetaData", "latitude"),
        "longitude": ("MetaData", "longitude"),
        "time": ("MetaData", "dateTime"),
    }

    export_to_ioda(ds, mapping, output_path)

    assert os.path.exists(output_path)

    # Verify by reading it back with our reader
    ds_back = monetio.load("ioda", files=output_path)

    assert "node" in ds_back.dims
    assert ds_back.sizes["node"] == 4  # 2 times * 2 sites

    assert "ozone_conc" in ds_back.variables
    assert "ozone_conc_error" in ds_back.variables
    assert "latitude" in ds_back.variables
    assert "time" in ds_back.variables
    assert "ozone_conc_qc" in ds_back.variables  # Auto-populated

    # Check a value
    # Stacking (time, site) -> (0,0), (0,1), (1,0), (1,1)
    np.testing.assert_allclose(ds_back.ozone_conc.values, [30.0, 31.0, 32.0, 33.0])

    # Check MetaData dateTime was converted back to strings in file but read as datetime
    assert np.issubdtype(ds_back.time.dtype, np.datetime64)


def test_ioda_read_multiple(tmp_path, mock_ioda_file):
    # Create a second file
    fn2 = tmp_path / "test_ioda_2.nc"
    with netCDF4.Dataset(fn2, "w") as nc:
        meta = nc.createGroup("MetaData")
        obs = nc.createGroup("ObsValue")
        meta.createDimension("nlocs", 1)
        obs.createDimension("nlocs", 1)

        lat = meta.createVariable("latitude", "f4", ("nlocs",))
        lat[:] = [60.0]
        dt = meta.createVariable("dateTime", str, ("nlocs",))
        dt[0] = "2023-01-02T00:00:00Z"
        o3 = obs.createVariable("ozone_conc", "f4", ("nlocs",))
        o3[:] = [60.0]

    ds = monetio.load("ioda", files=[mock_ioda_file, str(fn2)])
    assert ds.sizes["node"] == 6
    assert ds.latitude.values[-1] == 60.0
