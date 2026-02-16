from unittest.mock import patch

import pandas as pd
import pytest
import xarray as xr

from monetio.readers.nadp import NADPReader


@pytest.fixture
def dummy_nadp_file(tmp_path):
    fn = tmp_path / "NTN-All-w.csv"
    # NTN format: siteid, network, dateon, dateoff, ...
    # TX01, NTN, 2023-01-01, 2023-01-08, 1.0, 2.0
    content = "siteid,network,dateon,dateoff,mg,so4\nTX01,NTN,2023-01-01,2023-01-08,1.0,2.0\n"
    fn.write_text(content)
    return str(fn)


@pytest.fixture
def mock_meta():
    return pd.DataFrame({"siteid": ["TX01"], "latitude": [30.0], "longitude": [-100.0]})


def test_nadp_reader_eager(dummy_nadp_file, mock_meta):
    reader = NADPReader()

    with patch("pandas.read_csv") as mock_read:
        # First call is for the data file, second for metadata
        # But wait, driver calls reader_func which calls pd.read_csv
        # And get_monitor_df also calls pd.read_csv

        def side_effect(url, **kwargs):
            if "NTN-All-w.csv" in str(url):
                return pd.DataFrame(
                    {
                        "siteid": ["TX01"],
                        "network": ["NTN"],
                        "dateon": [pd.Timestamp("2023-01-01")],
                        "dateoff": [pd.Timestamp("2023-01-08")],
                        "mg": [1.0],
                        "so4": [2.0],
                    }
                )
            else:
                return mock_meta

        mock_read.side_effect = side_effect

        ds = reader.open_dataset(files=dummy_nadp_file, as_xarray=True, lazy=False)

    assert isinstance(ds, xr.Dataset)
    assert "time" in ds.coords
    assert "node" in ds.coords
    assert ds.sizes["node"] == 1
    assert ds.latitude.values[0] == 30.0


def test_nadp_reader_lazy(dummy_nadp_file, mock_meta):
    pytest.importorskip("dask")
    reader = NADPReader()

    with patch("pandas.read_csv") as mock_read:

        def side_effect(url, **kwargs):
            if "NTN-All-w.csv" in str(url):
                return pd.DataFrame(
                    {
                        "siteid": ["TX01"],
                        "network": ["NTN"],
                        "dateon": [pd.Timestamp("2023-01-01")],
                        "dateoff": [pd.Timestamp("2023-01-08")],
                        "mg": [1.0],
                        "so4": [2.0],
                    }
                )
            else:
                return mock_meta

        mock_read.side_effect = side_effect

        ds = reader.open_dataset(files=dummy_nadp_file, as_xarray=True, lazy=True)

    assert isinstance(ds, xr.Dataset)
    assert ds.mg.chunks is not None

    ds_eager = ds.compute()
    assert ds_eager.latitude.values[0] == 30.0
