import numpy as np
import pandas as pd
import pytest

from monetio.readers.ish_lite import ISHLiteReader


@pytest.fixture
def mock_ish_lite_files(tmp_path):
    # Site 1
    fn1 = tmp_path / "012345-67890-2023.gz"
    # year month day hour temp dew_pt_temp press wdir ws sky_condition precip_1hr precip_6hr
    content1 = "2023 01 01 00  100  50 10132  270   50 0 0 0\n"
    content1 += "2023 01 01 01  110  60 10135  280   60 0 0 0\n"

    # Site 2
    fn2 = tmp_path / "543210-09876-2023.gz"
    content2 = "2023 01 01 00  200  80 10132  270   50 0 0 0\n"
    content2 += "2023 01 01 01  220  90 10135  280   60 0 0 0\n"

    import gzip

    with gzip.open(fn1, "wb") as f:
        f.write(content1.encode())
    with gzip.open(fn2, "wb") as f:
        f.write(content2.encode())
    return [str(fn1), str(fn2)]


def test_ish_lite_multi_site_resample(mock_ish_lite_files, monkeypatch):
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

    reader = ISHLiteReader()

    # 1. Eager with resample
    # resample='h' shouldn't change data as it's already hourly
    ds = reader.open_dataset(
        files=mock_ish_lite_files, as_xarray=True, resample=True, window="h", lazy=False
    )

    assert "node" in ds.dims
    assert ds.sizes["node"] == 2
    assert ds.sizes["time"] == 2

    # Verify separate site data is preserved
    # Site 1 at 00Z should be 10.0
    # Site 2 at 00Z should be 20.0
    # Before the fix, they might have been averaged into 15.0 if they shared a time slot in a 1D array
    # But now they are separated by 'node' (siteid)

    # Identify which node is which siteid
    siteids = ds.siteid.values
    node_012345 = np.where(siteids == "01234567890")[0][0]
    node_543210 = np.where(siteids == "54321009876")[0][0]

    assert ds.temp.isel(time=0, node=node_012345).values == 10.0
    assert ds.temp.isel(time=0, node=node_543210).values == 20.0

    # Verify metadata is preserved
    assert ds.latitude.isel(node=node_012345).values == 40.0
    assert ds.latitude.isel(node=node_543210).values == 41.0


def test_ish_lite_multi_site_resample_lazy(mock_ish_lite_files, monkeypatch):
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

    reader = ISHLiteReader()

    # 2. Lazy with resample
    ds = reader.open_dataset(
        files=mock_ish_lite_files, as_xarray=True, resample=True, window="h", lazy=True
    )

    assert ds.temp.chunks is not None

    ds_computed = ds.compute()

    assert "node" in ds_computed.dims
    assert ds_computed.sizes["node"] == 2

    siteids = ds_computed.siteid.values
    node_012345 = np.where(siteids == "01234567890")[0][0]
    node_543210 = np.where(siteids == "54321009876")[0][0]

    assert ds_computed.temp.isel(time=0, node=node_012345).values == 10.0
    assert ds_computed.temp.isel(time=0, node=node_543210).values == 20.0


if __name__ == "__main__":
    pytest.main([__file__])
