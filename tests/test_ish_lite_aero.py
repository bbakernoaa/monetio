import os

import pandas as pd
import pytest

from monetio.readers.ish_lite import ISHLiteReader, read_ish_lite_file


def create_mock_ish_lite(path):
    """Create a mock ISH lite file."""
    # year month day hour temp dew_pt press wdir ws sky precip1 precip6
    data = [
        "2020 09 01 00    200    150  10130    180     50      0      0      0",
        "2020 09 01 01    210    160  10125    190     60      0      0      0",
        "2020 09 01 02    220    170  10120    200     70      0      0      0",
        "2020 09 01 03    230    180  10115    210     80      0      0      0",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(data))


def test_read_ish_lite_file(tmp_path):
    fn = str(tmp_path / "722244-00358-2020")
    create_mock_ish_lite(fn)

    df = read_ish_lite_file(fn)

    assert len(df) == 4
    assert "time" in df.columns
    assert df.siteid.iloc[0] == "72224400358"
    assert df.temp.iloc[0] == 20.0  # 200 / 10
    assert df.ws.iloc[0] == 5.0  # 50 / 10


@pytest.mark.parametrize("lazy", [False, True])
def test_ish_lite_resample_aero(tmp_path, lazy):
    # Setup mock files and history
    fn = str(tmp_path / "722244-00358-2020")
    create_mock_ish_lite(fn)

    reader = ISHLiteReader()

    # Mock build_urls and read_ish_history to use local file
    class MockISH:
        def __init__(self):
            self.history = pd.DataFrame(
                {
                    "station_id": ["72224400358"],
                    "usaf": ["722244"],
                    "wban": ["00358"],
                    "latitude": [38.98],
                    "longitude": [-76.92],
                    "ctry": ["US"],
                    "state": ["MD"],
                    "begin": [pd.Timestamp("2000-01-01")],
                    "end": [pd.Timestamp("2021-01-01")],
                }
            )
            self.dates = pd.to_datetime(["2020-09-01", "2020-09-02"])

        def read_ish_history(self):
            pass

        def subset_sites(self, **kwargs):
            return self.history

        def build_urls(self, **kwargs):
            return pd.DataFrame({"name": [fn]})

    import unittest.mock as mock

    with mock.patch("monetio.readers.ish_lite.ISH", return_value=MockISH()):
        ds = reader.open_dataset(
            dates=["2020-09-01", "2020-09-02"],
            site="72224400358",
            resample=True,
            window="2h",
            lazy=lazy,
            as_xarray=True,
        )
    if lazy:
        # Check that it's dask-backed
        assert ds.temp.chunks is not None
        ds = ds.compute()

    # After resampling 4 hours (00, 01, 02, 03) into 2h bins:
    # Bin 1 (00:00): mean of 00 and 01 -> (20 + 21)/2 = 20.5
    # Bin 2 (02:00): mean of 02 and 03 -> (22 + 23)/2 = 22.5

    assert len(ds.time) == 2
    assert ds.temp.sel(time="2020-09-01T00:00") == 20.5
    assert ds.temp.sel(time="2020-09-01T02:00") == 22.5

    # Check metadata preservation
    assert (ds.latitude == 38.98).all()
    assert (ds.node == "72224400358").all()
    assert "country" in ds.data_vars or "country" in ds.coords
