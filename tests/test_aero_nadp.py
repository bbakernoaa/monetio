from unittest.mock import patch

import pandas as pd
import pytest
import xarray as xr

from monetio.readers.nadp import NADPReader, read_nadp


@pytest.fixture
def mock_nadp_file(tmp_path):
    f = tmp_path / "nadp_test.csv"
    # Example columns for NTN: Real files often have an extra column at start or SiteID is at index 1.
    # The reader uses parse_dates=[2, 3] for NTN, which implies DateOn is at index 2.
    content = "ID,SiteID,DateOn,DateOff,Mg,flagMg,so4,flagso4\n"
    content += "1,MD99,2023-01-01,2023-01-08,0.5,,1.2,\n"
    content += "2,MD99,2023-01-08,2023-01-15,0.6,,1.3,\n"
    f.write_text(content)
    return str(f)


def test_read_nadp(mock_nadp_file):
    df = read_nadp(mock_nadp_file, network="ntn")
    assert len(df) == 2
    assert "time" in df.columns
    assert "time_off" in df.columns
    assert df["time"].iloc[0] == pd.Timestamp("2023-01-01")


@patch("monetio.readers.nadp.NADPReader.get_monitor_df")
def test_nadp_reader_eager(mock_get_monitor, mock_nadp_file):
    mock_get_monitor.return_value = pd.DataFrame(
        {"siteid": ["MD99"], "latitude": [39.0], "longitude": [-76.5]}
    )

    reader = NADPReader()
    ds = reader.open_dataset(files=mock_nadp_file, network="ntn", as_xarray=True, lazy=False)

    assert isinstance(ds, xr.Dataset)
    assert "mg" in ds.data_vars
    assert ds.sizes["node"] == 1
    assert ds.sizes["time"] == 2
    assert "Merged with NADP (ntn) station metadata" in ds.attrs["history"]


@patch("monetio.readers.nadp.NADPReader.get_monitor_df")
def test_nadp_reader_lazy(mock_get_monitor, mock_nadp_file):
    pytest.importorskip("dask")
    mock_get_monitor.return_value = pd.DataFrame(
        {"siteid": ["MD99"], "latitude": [39.0], "longitude": [-76.5]}
    )

    reader = NADPReader()
    ds = reader.open_dataset(files=mock_nadp_file, network="ntn", as_xarray=True, lazy=True)

    assert isinstance(ds, xr.Dataset)
    assert ds.mg.chunks is not None

    ds_eager = reader.open_dataset(files=mock_nadp_file, network="ntn", as_xarray=True, lazy=False)
    xr.testing.assert_allclose(ds.compute(), ds_eager)
