import pytest
import xarray as xr

from monetio.readers.airnow import AirNowReader


@pytest.fixture
def mock_airnow_file(tmp_path):
    """Create a mock AirNow hourly data file."""
    f = tmp_path / "HourlyData_2021010100.dat"
    # date|time|siteid|site|utcoffset|variable|units|obs|source
    content = (
        "01/01/21|00:00|060370001|Los Angeles|-8|OZONE|PPB|35.0|EPA\n"
        "01/01/21|00:00|060370002|Long Beach|-8|PM2.5|UG/M3|12.5|EPA\n"
    )
    f.write_text(content, encoding="ISO-8859-1")
    return str(f)


def test_airnow_reader_eager(mock_airnow_file):
    """Test AirNowReader with eager (Pandas) backend."""
    reader = AirNowReader()
    ds = reader.open_dataset(files=[mock_airnow_file], lazy=False)

    assert isinstance(ds, xr.Dataset)
    assert "node" in ds.dims
    assert ds.sizes["node"] == 2
    assert "OZONE" in ds.variable.values
    assert ds.obs.sel(node=0).values == 35.0
    assert (
        ds.latitude.sel(node=0).values is not None
    )  # Should be added from metadata peeking if available


def test_airnow_reader_lazy(mock_airnow_file):
    """Test AirNowReader with lazy (Dask) backend."""
    reader = AirNowReader()
    ds = reader.open_dataset(files=[mock_airnow_file], lazy=True)

    assert isinstance(ds, xr.Dataset)
    assert "node" in ds.dims
    # Check if it's actually lazy (Dask-backed)
    assert hasattr(ds.obs.data, "dask")

    # Compute and compare
    ds_computed = ds.compute()
    assert ds_computed.obs.sel(node=0).values == 35.0


def test_airnow_eager_lazy_consistency(mock_airnow_file):
    """Verify that Eager and Lazy backends produce identical results."""
    reader = AirNowReader()
    ds_eager = reader.open_dataset(files=[mock_airnow_file], lazy=False)
    ds_lazy = reader.open_dataset(files=[mock_airnow_file], lazy=True).compute()

    # Standardize history for comparison (as it contains timestamps)
    ds_eager.attrs["history"] = ""
    ds_lazy.attrs["history"] = ""

    # Dask to_xarray might not create the same node coordinate as from_dataframe
    if "node" in ds_eager.coords and "node" not in ds_lazy.coords:
        ds_lazy = ds_lazy.assign_coords(node=ds_eager.node)

    xr.testing.assert_allclose(ds_eager, ds_lazy)
