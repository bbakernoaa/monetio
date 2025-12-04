import pytest
from unittest.mock import MagicMock
import xarray as xr
import pandas as pd
import numpy as np
from io import BytesIO

from monetio.models import icap_mme

# Mock response content for a minimal netCDF file
def create_mock_nc_content():
    ds = xr.Dataset(
        {
            "dust_aod_mean": (("time", "lat", "lon"), np.random.rand(4, 2, 2)),
        },
        coords={
            "time": pd.date_range("2023-08-01", periods=4, freq="6h"),
            "lat": [0, 1],
            "lon": [0, 1],
        }
    )
    # Use scipy engine to write NETCDF3 which works well with in-memory bytes
    return ds.to_netcdf(engine="scipy", format="NETCDF3_64BIT")

@pytest.fixture
def mock_requests(monkeypatch):
    mock_get = MagicMock()
    mock_head = MagicMock()

    # Mock HEAD to return 200 OK
    mock_head.return_value.status_code = 200

    # Mock GET to return content
    mock_get.return_value.status_code = 200
    mock_get.return_value.content = create_mock_nc_content()

    # Support for stream=True
    # The 'retrieve' function calls requests.get(..., stream=True)
    # Then it reads r.content
    # When stream=True, iterating content is different, but r.content property should still work if mocked correctly.
    # MagicMock handles property access via attributes.

    monkeypatch.setattr("requests.head", mock_head)
    monkeypatch.setattr("requests.get", mock_get)
    return mock_get, mock_head

def test_open_dataset_bad_date(mock_requests):
    mock_get, mock_head = mock_requests
    mock_head.return_value.status_code = 404

    with pytest.raises(ValueError, match="File does not exist"):
        icap_mme.open_dataset("1990-08-01")

def test_open_dataset_invalid_param():
    date = "2019-08-01"

    with pytest.raises(ValueError, match="Invalid input for 'product'"):
        icap_mme.open_dataset(date, product="asdf")
        icap_mme.open_mfdataset([date], product="asdf")

    with pytest.raises(ValueError, match="Invalid input for 'data_var'"):
        icap_mme.open_dataset(date, data_var="asdf")
        icap_mme.open_mfdataset([date], data_var="asdf")


@pytest.mark.parametrize(
    "date,product,data_var",
    [
        ("2019-08-01", "MME", "totaldustaod550"),
        ("2024-02-01", "C4", "dustaod550"),
    ],
)
def test_open_dataset(tmp_path, monkeypatch, mock_requests, date, product, data_var):
    ds = icap_mme.open_dataset(date, product=product, data_var=data_var, download=False)
    assert set(ds.dims) == {"time", "lat", "lon"}

    monkeypatch.chdir(tmp_path)
    ds_dl = icap_mme.open_dataset(date, product=product, data_var=data_var, download=True)
    assert len(sorted(tmp_path.glob("*.nc"))) == 1
    assert set(ds_dl.dims) == {"time", "lat", "lon"}

    # Equality check
    assert ds_dl.equals(ds)


def test_open_mfdataset(tmp_path, monkeypatch, mock_requests):
    dates = ["2023-08-01", "2023-08-02"]
    product = "C4"
    data_var = "dustaod550"

    ds = icap_mme.open_mfdataset(dates, product=product, data_var=data_var, download=False)
    assert set(ds.dims) == {"time", "lat", "lon"}

    # Check concatenation happened (time dim size should be larger than single file)
    # The mock content has 4 time steps. We query 2 dates.
    # icap_mme.open_mfdataset iterates urls and concatenates.
    # 2 dates -> 2 urls -> 2 * 4 = 8 time steps
    assert ds.sizes['time'] == 8

    monkeypatch.chdir(tmp_path)
    ds_dl = icap_mme.open_mfdataset(dates, product=product, data_var=data_var, download=True)
    assert len(sorted(tmp_path.glob("*.nc"))) == 2
    assert set(ds_dl.dims) == {"time", "lat", "lon"}
    assert ds_dl.sizes['time'] == 8

    assert ds_dl.equals(ds)
