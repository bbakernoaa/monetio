from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monetio.readers.crn import CRNReader, read_crn


def test_parse_yyyymmdd_hhmm_logic():
    from monetio.readers.time_utils import parse_yyyymmdd_hhmm

    yyyymmdd = np.array([20230101, 20231231])
    hhmm = np.array([0, 2359])
    res = parse_yyyymmdd_hhmm(yyyymmdd, hhmm)
    expected = pd.to_datetime(["2023-01-01 00:00:00", "2023-12-31 23:59:00"]).values.astype(
        "datetime64[ns]"
    )
    np.testing.assert_array_equal(res, expected)

    hhmmss = np.array([100, 123045])
    res2 = parse_yyyymmdd_hhmm(yyyymmdd, hhmmss)
    # 100 is treated as HHMM if max is < 10000, but here 123045 makes it HHMMSS
    # 000100 -> 00:01:00
    # 123045 -> 12:30:45
    expected2 = pd.to_datetime(["2023-01-01 00:01:00", "2023-12-31 12:30:45"]).values.astype(
        "datetime64[ns]"
    )
    np.testing.assert_array_equal(res2, expected2)


@pytest.fixture
def mock_crn_file(tmp_path):
    d = tmp_path / "crn_data"
    d.mkdir()
    f = d / "CRNH0203-2023-MD_Test_Site.txt"
    # HCOLS: WBANNO UTC_DATE UTC_TIME LST_DATE LST_TIME CRX_VN LONGITUDE LATITUDE T_CALC ...
    # Just a few columns for testing
    content = "64758 20230101 0000 20221231 1900 1.0 -76.5 39.0 10.5 10.0 11.0 9.5 0.0 500.0 0 550.0 0 450.0 0 H 15.0 0 16.0 0 14.0 0 80.0 0 0.2 0.2 0.2 0.2 0.2 12.0 12.0 12.0 12.0 12.0\n"
    content += "64758 20230101 0100 20221231 2000 1.0 -76.5 39.0 10.6 10.1 11.1 9.6 0.0 501.0 0 551.0 0 451.0 0 H 15.1 0 16.1 0 14.1 0 81.0 0 0.2 0.2 0.2 0.2 0.2 12.1 12.1 12.1 12.1 12.1\n"
    f.write_text(content)
    return str(f)


def test_read_crn(mock_crn_file):
    df = read_crn(mock_crn_file)
    assert len(df) == 2
    assert "time" in df.columns
    assert "time_local" in df.columns
    assert df["time"].iloc[0] == pd.Timestamp("2023-01-01 00:00:00")
    assert df["time_local"].iloc[0] == pd.Timestamp("2022-12-31 19:00:00")


@patch("monetio.readers.crn.CRNReader.get_monitor_df")
def test_crn_reader_eager(mock_get_monitor, mock_crn_file):
    # Mock monitor DF
    monitor_df = pd.DataFrame(
        {
            "WBAN": ["64758"],
            "STATE": ["MD"],
            "LOCATION": ["Test Site"],
            "VECTOR": ["Test Vector"],
            "LATITUDE": [39.0],
            "LONGITUDE": [-76.5],
            "NETWORK": ["USCRN"],
            "GMT_OFFSET": [-5.0],
        }
    )
    mock_get_monitor.return_value = monitor_df

    reader = CRNReader()
    # Test with files provided directly to avoid build_urls network call
    ds = reader.open_dataset(files=mock_crn_file, as_xarray=True, lazy=False)

    assert isinstance(ds, xr.Dataset)
    assert "t_calc" in ds.data_vars
    assert ds.sizes["node"] == 1
    assert ds.sizes["time"] == 2
    # Check if history is updated
    assert "Merged with CRN station metadata" in ds.attrs["history"]


@patch("monetio.readers.crn.CRNReader.get_monitor_df")
def test_crn_reader_lazy(mock_get_monitor, mock_crn_file):
    pytest.importorskip("dask")
    # Mock monitor DF
    monitor_df = pd.DataFrame(
        {
            "WBAN": ["64758"],
            "STATE": ["MD"],
            "LOCATION": ["Test Site"],
            "VECTOR": ["Test Vector"],
            "LATITUDE": [39.0],
            "LONGITUDE": [-76.5],
            "NETWORK": ["USCRN"],
            "GMT_OFFSET": [-5.0],
        }
    )
    mock_get_monitor.return_value = monitor_df

    reader = CRNReader()
    ds = reader.open_dataset(files=mock_crn_file, as_xarray=True, lazy=True)

    assert isinstance(ds, xr.Dataset)
    # Check if underlying data is dask
    assert ds.t_calc.chunks is not None

    # Compute and compare with eager
    ds_eager = reader.open_dataset(files=mock_crn_file, as_xarray=True, lazy=False)
    xr.testing.assert_allclose(ds.compute(), ds_eager)


def test_crn_build_urls_optimization():
    reader = CRNReader()
    dates = pd.to_datetime(["2023-01-01"])

    # Mock monitors
    monitor_df = pd.DataFrame(
        {
            "STATE": ["MD"],
            "LOCATION": ["Test Site"],
            "VECTOR": ["Test"],
            "LATITUDE": [39.0],
            "LONGITUDE": [-76.5],
        }
    )

    with patch("monetio.readers.crn.CRNReader.get_monitor_df", return_value=monitor_df):
        with patch("monetio.readers.crn.FileUtility.get_fs") as mock_get_fs:
            mock_fs = MagicMock()
            mock_get_fs.return_value = mock_fs

            # Mock ls to return our expected file
            mock_fs.ls.return_value = [
                "https://www1.ncdc.noaa.gov/pub/data/uscrn/products/hourly02/2023/CRNH0203-2023-MD_Test_Site_Test.txt"
            ]

            urls, fnames = reader.build_urls(dates)

            assert len(urls) == 1
            assert "CRNH0203-2023-MD_Test_Site_Test.txt" in fnames
            # Ensure ls was called once (optimization check)
            assert mock_fs.ls.call_count == 1
            # Ensure exists was NOT called if ls worked
            assert mock_fs.exists.call_count == 0


def test_crn_build_urls_fallback():
    reader = CRNReader()
    dates = pd.to_datetime(["2023-01-01"])

    # Mock monitors
    monitor_df = pd.DataFrame(
        {
            "STATE": ["MD"],
            "LOCATION": ["Test Site"],
            "VECTOR": ["Test"],
            "LATITUDE": [39.0],
            "LONGITUDE": [-76.5],
        }
    )

    with patch("monetio.readers.crn.CRNReader.get_monitor_df", return_value=monitor_df):
        with patch("monetio.readers.crn.FileUtility.get_fs") as mock_get_fs:
            mock_fs = MagicMock()
            mock_get_fs.return_value = mock_fs

            # Mock ls to FAIL
            mock_fs.ls.side_effect = Exception("ls not supported")
            # Mock exists to return True
            mock_fs.exists.return_value = True

            urls, fnames = reader.build_urls(dates)

            assert len(urls) == 1
            # Ensure exists was called as fallback
            assert mock_fs.exists.call_count == 1
