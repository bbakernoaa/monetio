from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from monetio.readers.ish import ISHReader, read_ish_file


def test_read_ish_file_logic(tmp_path):
    # Create a mock ISH file
    line1 = (
        "0054"
        + "72224400358"
        + "20200901"
        + "0000"
        + "4"
        + "+38941"
        + "-076952"
        + "FM-12"
        + "+0020"
        + "KADW "
        + "9999"
        + "220"
        + "1"
        + "V"
        + "0040"
        + "1"
        + "99999"
        + "1"
        + "9"
        + "9"
        + "030000"
        + "1"
        + "9"
        + "9"
        + "+0256"
        + "1"
        + "+0184"
        + "1"
        + "10185"
        + "1"
    )
    line2 = (
        "0054"
        + "72224400358"
        + "20200901"
        + "0100"
        + "4"
        + "+38941"
        + "-076952"
        + "FM-12"
        + "+0020"
        + "KADW "
        + "9999"
        + "230"
        + "1"
        + "V"
        + "0050"
        + "1"
        + "99999"
        + "1"
        + "9"
        + "9"
        + "035000"
        + "1"
        + "9"
        + "9"
        + "+0260"
        + "1"
        + "+0190"
        + "1"
        + "10180"
        + "1"
    )

    fn = tmp_path / "722244-00358-2020"
    with open(fn, "w") as f:
        f.write(line1 + "\n")
        f.write(line2 + "\n")

    df = read_ish_file(str(fn))

    assert len(df) == 2
    assert df.time.iloc[0] == pd.Timestamp("2020-09-01 00:00:00")
    assert df.time.iloc[1] == pd.Timestamp("2020-09-01 01:00:00")
    assert df.station_id.iloc[0] == "72224400358"
    assert df.t.iloc[0] == 25.6
    assert df.t.iloc[1] == 26.0


def test_ish_eager_lazy_consistency(tmp_path):
    # Setup mock data
    mock_history = pd.DataFrame(
        {
            "station_id": ["72224400358"],
            "usaf": ["722244"],
            "wban": ["00358"],
            "latitude": [38.941],
            "longitude": [-76.952],
            "ctry": ["US"],
            "state": ["MD"],
            "station name": ["Mock Station"],
            "elev(m)": [20.0],
            "begin": [pd.Timestamp("1970-01-01")],
            "end": [pd.Timestamp("2025-01-01")],
        }
    )

    line1 = (
        "0054"
        + "72224400358"
        + "20200901"
        + "0000"
        + "4"
        + "+38941"
        + "-076952"
        + "FM-12"
        + "+0020"
        + "KADW "
        + "9999"
        + "220"
        + "1"
        + "V"
        + "0040"
        + "1"
        + "99999"
        + "1"
        + "9"
        + "9"
        + "030000"
        + "1"
        + "9"
        + "9"
        + "+0256"
        + "1"
        + "+0184"
        + "1"
        + "10185"
        + "1"
    )
    line2 = (
        "0054"
        + "72224400358"
        + "20200901"
        + "0100"
        + "4"
        + "+38941"
        + "-076952"
        + "FM-12"
        + "+0020"
        + "KADW "
        + "9999"
        + "230"
        + "1"
        + "V"
        + "0050"
        + "1"
        + "99999"
        + "1"
        + "9"
        + "9"
        + "035000"
        + "1"
        + "9"
        + "9"
        + "+0260"
        + "1"
        + "+0190"
        + "1"
        + "10180"
        + "1"
    )

    fn = tmp_path / "722244-00358-2020"
    with open(fn, "w") as f:
        f.write(line1 + "\n")
        f.write(line2 + "\n")

    reader = ISHReader()

    def side_effect(self, dates=None):
        self.history = mock_history

    with patch("monetio.readers.ish.ISH.read_ish_history", autospec=True, side_effect=side_effect):
        # Eager path
        ds_eager = reader.open_dataset(files=str(fn), as_xarray=True, lazy=False, resample=False)

        # Lazy path
        ds_lazy = reader.open_dataset(files=str(fn), as_xarray=True, lazy=True, resample=False)

    assert isinstance(ds_eager.t.data, np.ndarray)
    try:
        import dask.array as da

        assert isinstance(ds_lazy.t.data, da.Array)
    except ImportError:
        pass

    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())
    assert "Read ISH data" in ds_eager.attrs["history"]
    assert "Read ISH data" in ds_lazy.attrs["history"]


def test_ish_resampling_logic(tmp_path):
    # Setup mock data
    mock_history = pd.DataFrame(
        {
            "station_id": ["72224400358"],
            "usaf": ["722244"],
            "wban": ["00358"],
            "latitude": [38.941],
            "longitude": [-76.952],
            "ctry": ["US"],
            "state": ["MD"],
            "station name": ["Mock Station"],
            "elev(m)": [20.0],
            "begin": [pd.Timestamp("1970-01-01")],
            "end": [pd.Timestamp("2025-01-01")],
        }
    )

    # Create 3 hours of data
    line1 = (
        "0054"
        + "72224400358"
        + "20200901"
        + "0000"
        + "4"
        + "+38941"
        + "-076952"
        + "FM-12"
        + "+0020"
        + "KADW "
        + "9999"
        + "220"
        + "1"
        + "V"
        + "0040"
        + "1"
        + "99999"
        + "1"
        + "9"
        + "9"
        + "030000"
        + "1"
        + "9"
        + "9"
        + "+0200"
        + "1"
        + "+0100"
        + "1"
        + "10000"
        + "1"
    )
    line2 = (
        "0054"
        + "72224400358"
        + "20200901"
        + "0100"
        + "4"
        + "+38941"
        + "-076952"
        + "FM-12"
        + "+0020"
        + "KADW "
        + "9999"
        + "220"
        + "1"
        + "V"
        + "0040"
        + "1"
        + "99999"
        + "1"
        + "9"
        + "9"
        + "030000"
        + "1"
        + "9"
        + "9"
        + "+0210"
        + "1"
        + "+0110"
        + "1"
        + "10010"
        + "1"
    )
    line3 = (
        "0054"
        + "72224400358"
        + "20200901"
        + "0200"
        + "4"
        + "+38941"
        + "-076952"
        + "FM-12"
        + "+0020"
        + "KADW "
        + "9999"
        + "220"
        + "1"
        + "V"
        + "0040"
        + "1"
        + "99999"
        + "1"
        + "9"
        + "9"
        + "030000"
        + "1"
        + "9"
        + "9"
        + "+0220"
        + "1"
        + "+0120"
        + "1"
        + "10020"
        + "1"
    )

    fn = tmp_path / "722244-00358-2020"
    with open(fn, "w") as f:
        f.write(line1 + "\n")
        f.write(line2 + "\n")
        f.write(line3 + "\n")

    reader = ISHReader()

    def side_effect(self, dates=None):
        self.history = mock_history

    with patch("monetio.readers.ish.ISH.read_ish_history", autospec=True, side_effect=side_effect):
        # Resample to 3h
        ds = reader.open_dataset(
            files=str(fn), as_xarray=True, lazy=False, resample=True, window="3h"
        )

    assert len(ds.time) == 1
    assert ds.t.values[0, 0] == 21.0  # 2D (time, node)
    assert ds.p.values[0, 0] == 1001.0


def test_ish_multi_site_resampling(tmp_path):
    # Setup mock data for two sites
    mock_history = pd.DataFrame(
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

    # Site 1 data
    s1_line1 = (
        "0054"
        + "72224400358"
        + "20200901"
        + "0000"
        + "4"
        + "+38941"
        + "-076952"
        + "FM-12"
        + "+0020"
        + "KADW "
        + "9999"
        + "220"
        + "1"
        + "V"
        + "0040"
        + "1"
        + "99999"
        + "1"
        + "9"
        + "9"
        + "030000"
        + "1"
        + "9"
        + "9"
        + "+0200"
        + "1"
        + "+0100"
        + "1"
        + "10000"
        + "1"
    )
    s1_line2 = (
        "0054"
        + "72224400358"
        + "20200901"
        + "0100"
        + "4"
        + "+38941"
        + "-076952"
        + "FM-12"
        + "+0020"
        + "KADW "
        + "9999"
        + "220"
        + "1"
        + "V"
        + "0040"
        + "1"
        + "99999"
        + "1"
        + "9"
        + "9"
        + "030000"
        + "1"
        + "9"
        + "9"
        + "+0210"
        + "1"
        + "+0110"
        + "1"
        + "10010"
        + "1"
    )

    # Site 2 data
    s2_line1 = (
        "0054"
        + "99999912345"
        + "20200901"
        + "0000"
        + "4"
        + "+40000"
        + "-080000"
        + "FM-12"
        + "+0030"
        + "XXXX "
        + "9999"
        + "220"
        + "1"
        + "V"
        + "0040"
        + "1"
        + "99999"
        + "1"
        + "9"
        + "9"
        + "030000"
        + "1"
        + "9"
        + "9"
        + "+0300"
        + "1"
        + "+0200"
        + "1"
        + "10100"
        + "1"
    )
    s2_line2 = (
        "0054"
        + "99999912345"
        + "20200901"
        + "0100"
        + "4"
        + "+40000"
        + "-080000"
        + "FM-12"
        + "+0030"
        + "XXXX "
        + "9999"
        + "220"
        + "1"
        + "V"
        + "0040"
        + "1"
        + "99999"
        + "1"
        + "9"
        + "9"
        + "030000"
        + "1"
        + "9"
        + "9"
        + "+0310"
        + "1"
        + "+0210"
        + "1"
        + "10110"
        + "1"
    )

    fn1 = tmp_path / "722244-00358-2020"
    fn2 = tmp_path / "999999-12345-2020"
    with open(fn1, "w") as f:
        f.write(s1_line1 + "\n" + s1_line2 + "\n")
    with open(fn2, "w") as f:
        f.write(s2_line1 + "\n" + s2_line2 + "\n")

    reader = ISHReader()

    def side_effect(self, dates=None):
        self.history = mock_history

    with patch("monetio.readers.ish.ISH.read_ish_history", autospec=True, side_effect=side_effect):
        # Resample to 2h
        ds = reader.open_dataset(
            files=[str(fn1), str(fn2)], as_xarray=True, lazy=False, resample=True, window="2h"
        )

    assert len(ds.time) == 1
    assert len(ds.node) == 2

    # Identify indices
    idx1 = np.where(ds.siteid.values == "72224400358")[0][0]
    idx2 = np.where(ds.siteid.values == "99999912345")[0][0]

    assert ds.t.values[0, idx1] == 20.5  # Average of 20, 21
    assert ds.t.values[0, idx2] == 30.5  # Average of 30, 31
    assert ds.state.values[idx1] == "MD"
    assert ds.state.values[idx2] == "PA"
