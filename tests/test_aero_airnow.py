import pytest
import xarray as xr
from monetio.readers.airnow import AirNowReader


@pytest.fixture
def mock_airnow_file(tmp_path):
    """Create a mock AirNow file."""
    f = tmp_path / "HourlyData_2021010100.dat"
    content = "01/01/21|00:00|060371103|SiteName|0|OZONE|PPB|42.0|Source\n"
    f.write_text(content, encoding="ISO-8859-1")
    return str(f)


def test_airnow_reader_eager(mock_airnow_file):
    """Test AirNowReader with eager loading."""
    reader = AirNowReader()
    ds = reader.open_dataset(files=[mock_airnow_file], lazy=False)

    assert isinstance(ds, xr.Dataset)
    assert "node" in ds.dims
    assert ds.node.size == 1
    assert "obs" in ds.data_vars
    assert ds.obs.values[0] == 42.0
    assert "UGRID-1.0" in ds.attrs["Conventions"]
    assert "history" in ds.attrs


def test_airnow_reader_lazy(mock_airnow_file):
    """Test AirNowReader with lazy loading."""
    reader = AirNowReader()
    ds = reader.open_dataset(files=[mock_airnow_file], lazy=True)

    assert isinstance(ds, xr.Dataset)
    assert "node" in ds.dims
    # Check if data is dask-backed
    assert ds.obs.chunks is not None

    # Compute and verify
    ds_computed = ds.compute()
    assert ds_computed.obs.values[0] == 42.0
    assert "history" in ds.attrs


def test_airnow_consistency(mock_airnow_file):
    """Assert eager and lazy results are identical."""
    reader = AirNowReader()
    ds_eager = reader.open_dataset(files=[mock_airnow_file], lazy=False)
    ds_lazy = reader.open_dataset(files=[mock_airnow_file], lazy=True).compute()

    xr.testing.assert_allclose(ds_eager, ds_lazy)
    assert ds_eager.attrs["history"] != ""
    assert ds_lazy.attrs["history"] != ""
