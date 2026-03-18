from unittest.mock import patch

import pandas as pd
import pytest
import xarray as xr

from monetio.readers.aeronet import AERONETReader
from monetio.readers.improve import IMPROVEReader
from monetio.readers.pams import PAMSReader


@pytest.fixture
def mock_improve_file(tmp_path):
    f = tmp_path / "improve_test.txt"
    content = "Metadata lines...\nData\n"
    content += "SiteCode\tDate\tEPACode\tVal\tState\tParamCode\tUnit\n"
    content += "TEST1\t2023-01-01\t123456789\t10.0\tMD\tPM25\tug/m3\n"
    f.write_text(content)
    return str(f)


def test_improve_reader_eager(mock_improve_file):
    reader = IMPROVEReader()
    # Mock monitor file read
    with patch("monetio.readers.epa_utils.read_monitor_file") as mock_mon:
        mock_mon.return_value = pd.DataFrame(
            {"siteid": ["123456789"], "latitude": [39.0], "longitude": [-76.5]}
        )
        ds = reader.open_dataset(files=mock_improve_file, add_meta=True, as_xarray=True, lazy=False)
        assert isinstance(ds, xr.Dataset)
        assert "PM25" in ds.data_vars
        assert ds.sizes["time"] == 1
        assert "Merged with IMPROVE station metadata" in ds.attrs["history"]


def test_improve_reader_lazy(mock_improve_file):
    pytest.importorskip("dask")
    reader = IMPROVEReader()
    with patch("monetio.readers.epa_utils.read_monitor_file") as mock_mon:
        mock_mon.return_value = pd.DataFrame(
            {"siteid": ["123456789"], "latitude": [39.0], "longitude": [-76.5]}
        )
        ds = reader.open_dataset(files=mock_improve_file, add_meta=True, as_xarray=True, lazy=True)
        assert ds.PM25.chunks is not None
        ds_eager = reader.open_dataset(
            files=mock_improve_file, add_meta=True, as_xarray=True, lazy=False
        )
        xr.testing.assert_allclose(ds.compute(), ds_eager)


@pytest.fixture
def mock_pams_file(tmp_path):
    f = tmp_path / "pams_test.json"
    data = {
        "Data": [
            {
                "state_code": "24",
                "county_code": "001",
                "site_number": "0001",
                "date_gmt": "2023-01-01",
                "time_gmt": "00:00",
                "sample_measurement": 10.0,
                "units_of_measure": "Parts per billion",
                "latitude": 39.0,
                "longitude": -76.5,
            }
        ]
    }
    import json

    f.write_text(json.dumps(data))
    return str(f)


def test_pams_reader_eager(mock_pams_file):
    reader = PAMSReader()
    ds = reader.open_dataset(files=mock_pams_file, as_xarray=True, lazy=False)
    assert isinstance(ds, xr.Dataset)
    assert ds.obs.attrs["units"] == "ppb"
    assert ds.sizes["time"] == 1


def test_pams_reader_lazy(mock_pams_file):
    pytest.importorskip("dask")
    reader = PAMSReader()
    ds = reader.open_dataset(files=mock_pams_file, as_xarray=True, lazy=True)
    # Underlying data should be dask
    assert ds.obs.chunks is not None
    assert ds.obs.attrs["units"] == "ppb"


@patch("monetio.readers.aeronet.FileUtility.get_fs")
def test_aeronet_reader_instantiation(mock_get_fs):
    reader = AERONETReader()
    assert reader is not None
