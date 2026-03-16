from unittest.mock import patch

import pandas as pd
import pytest
import xarray as xr

from monetio.readers.cems import CEMSReader, read_cems
from monetio.readers.icap_mme import ICAPMMEReader


@pytest.fixture
def mock_cems_file(tmp_path):
    f = tmp_path / "cems_test.csv"
    # Columns: Facility Name, ORISPL, Date, Hour, Latitude, Longitude, ...
    content = (
        "Facility Name,ORISPL,Date,Hour,Latitude,Longitude,SO2 (lbs),NOx (lbs),CO2 (short tons)\n"
    )
    content += "Test Plant,1234,2023-01-01,0,39.0,-76.5,10.0,5.0,100.0\n"
    content += "Test Plant,1234,2023-01-01,1,39.0,-76.5,11.0,6.0,101.0\n"
    f.write_text(content)
    return str(f)


def test_read_cems(mock_cems_file):
    df = read_cems(mock_cems_file)
    assert len(df) == 2
    assert "time" in df.columns
    assert df["time"].iloc[0] == pd.Timestamp("2023-01-01 00:00:00")
    assert df["time"].iloc[1] == pd.Timestamp("2023-01-01 01:00:00")
    assert df["siteid"].iloc[0] == "1234"


def test_cems_reader_eager(mock_cems_file):
    reader = CEMSReader()
    ds = reader.open_dataset(files=mock_cems_file, as_xarray=True, lazy=False)
    assert isinstance(ds, xr.Dataset)
    assert "so2_lbs" in ds.data_vars
    assert ds.sizes["time"] == 2
    assert ds.sizes["node"] == 1


def test_cems_reader_lazy(mock_cems_file):
    pytest.importorskip("dask")
    reader = CEMSReader()
    ds = reader.open_dataset(files=mock_cems_file, as_xarray=True, lazy=True)
    assert ds.so2_lbs.chunks is not None
    ds_eager = reader.open_dataset(files=mock_cems_file, as_xarray=True, lazy=False)
    xr.testing.assert_allclose(ds.compute(), ds_eager)


@patch("monetio.readers.icap_mme.FileUtility.get_fs")
def test_icap_mme_reader_mock(mock_get_fs):
    # Mocking xarray.open_mfdataset is hard, so we just test build_urls
    from monetio.readers.icap_mme import build_urls

    dates = pd.to_datetime(["2024-02-01 00:00:00"])
    urls, fnames = build_urls(dates, filetype="MMC", data_var="dustaod550")
    assert isinstance(urls, list)
    assert len(urls) == 1
    assert "icap_2024020100_MMC_dustaod550.nc" in urls[0]
    assert "icap_2024020100_MMC_dustaod550.nc" == fnames[0]


def test_icap_mme_reader_instantiation():
    reader = ICAPMMEReader()
    assert reader is not None
