import gzip
from unittest.mock import patch

import pandas as pd
import pytest
import xarray as xr

from monetio.readers.ish import ISHReader, read_ish_file
from monetio.readers.ish_lite import ISHLiteReader


@pytest.fixture
def mock_history():
    return pd.DataFrame(
        {
            "station_id": ["72224400358", "99999912345"],
            "usaf": ["722244", "999999"],
            "wban": ["00358", "12345"],
            "latitude": [38.9, 40.0],
            "longitude": [-76.9, -80.0],
            "ctry": ["US", "US"],
            "state": ["MD", "PA"],
            "station name": ["Site 1", "Site 2"],
            "elev(m)": [20.0, 30.0],
            "begin": [pd.Timestamp("1970-01-01"), pd.Timestamp("1970-01-01")],
            "end": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")],
        }
    )


def test_read_ish_file_logic(tmp_path):
    line1 = "0054722244003582020090100004+38941-076952FM-12+0020KADW 99992201V0040199999199030000199+02561+01841101851"
    fn = tmp_path / "722244-00358-2020"
    fn.write_text(line1 + "\n")
    df = read_ish_file(str(fn))
    assert len(df) == 1
    assert df.time.iloc[0] == pd.Timestamp("2020-09-01 00:00:00")
    assert df.t.iloc[0] == 25.6


def test_ish_eager_lazy_consistency(tmp_path, mock_history):
    line1 = "0054722244003582020090100004+38941-076952FM-12+0020KADW 99992201V0040199999199030000199+02561+01841101851"
    fn = tmp_path / "722244-00358-2020"
    fn.write_text(line1 + "\n")
    reader = ISHReader()

    def side_effect(self, dates=None):
        self.history = mock_history

    with patch("monetio.readers.ish.ISH.read_ish_history", autospec=True, side_effect=side_effect):
        ds_eager = reader.open_dataset(files=str(fn), as_xarray=True, lazy=False, resample=False)
        ds_lazy = reader.open_dataset(files=str(fn), as_xarray=True, lazy=True, resample=False)
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())


def test_ish_lite_multi_site_resample(monkeypatch, tmp_path):
    def mock_read_history(self, dates=None):
        self.history = pd.DataFrame(
            {
                "usaf": ["012345", "543210"],
                "wban": ["67890", "09876"],
                "latitude": [40.0, 41.0],
                "longitude": [-80.0, -81.0],
                "station_id": ["01234567890", "54321009876"],
                "ctry": ["US", "US"],
                "state": ["PA", "OH"],
                "station name": ["Test Station 1", "Test Station 2"],
                "elev(m)": [100.0, 200.0],
                "begin": [pd.to_datetime("2020-01-01"), pd.to_datetime("2020-01-01")],
                "end": [pd.to_datetime("2025-01-01"), pd.to_datetime("2025-01-01")],
            }
        )

    import monetio.readers.ish as ish

    monkeypatch.setattr(ish.ISH, "read_ish_history", mock_read_history)
    fn1 = tmp_path / "012345-67890-2023.gz"
    with gzip.open(fn1, "wb") as f:
        f.write(b"2023 01 01 00  100  50 10132  270   50 0 0 0\n")
    fn2 = tmp_path / "543210-09876-2023.gz"
    with gzip.open(fn2, "wb") as f:
        f.write(b"2023 01 01 00  200  80 10132  270   50 0 0 0\n")
    reader = ISHLiteReader()
    ds = reader.open_dataset(
        files=[str(fn1), str(fn2)], as_xarray=True, resample=True, window="h", lazy=False
    )
    assert ds.sizes["node"] == 2


@pytest.mark.network
def test_ish_network():
    dates = pd.date_range("2020-09-01", "2020-09-01 01:00", freq="h")
    site = "72224400358"
    df = ISHReader().open_dataset(dates=dates, site=site, as_xarray=False)
    assert not df.empty


@pytest.mark.network
def test_ish_lite_network():
    dates = pd.date_range("2020-09-01", "2020-09-01 01:00", freq="h")
    site = "72224400358"
    df = ISHLiteReader().open_dataset(dates=dates, site=site, as_xarray=False)
    assert not df.empty
