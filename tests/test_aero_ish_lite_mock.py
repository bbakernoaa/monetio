import pandas as pd
import pytest
import xarray as xr

from monetio.readers.ish_lite import ISHLiteReader


@pytest.fixture
def mock_ish_lite_file(tmp_path):
    # USAF-WBAN-YEAR.gz
    fn = tmp_path / "012345-67890-2023.gz"
    # year month day hour temp dew_pt_temp press wdir ws sky_condition precip_1hr precip_6hr
    # Content is space separated, 12 columns
    content = "2023 01 01 00  100  50 10132  270   50 0 0 0\n"
    content += "2023 01 01 01  110  60 10135  280   60 0 0 0\n"
    import gzip

    with gzip.open(fn, "wb") as f:
        f.write(content.encode())
    return str(fn)


def test_ish_lite_eager_vs_lazy_local(mock_ish_lite_file, monkeypatch):
    def mock_read_history(self, dates=None):
        self.history = pd.DataFrame(
            {
                "usaf": ["012345"],
                "wban": ["67890"],
                "latitude": [40.0],
                "longitude": [-80.0],
                "station_id": ["01234567890"],
                "ctry": ["US"],
                "state": ["PA"],
                "station name": ["Test Station"],
                "elev(m)": [100.0],
                "begin": [pd.to_datetime("2020-01-01")],
                "end": [pd.to_datetime("2025-01-01")],
            }
        )

    import monetio.readers.ish as ish

    monkeypatch.setattr(ish.ISH, "read_ish_history", mock_read_history)

    reader = ISHLiteReader()

    # 1. Eager
    ds_eager = reader.open_dataset(files=mock_ish_lite_file, as_xarray=True, lazy=False)

    # 2. Lazy
    ds_lazy = reader.open_dataset(files=mock_ish_lite_file, as_xarray=True, lazy=True)

    # Check that ds_lazy is indeed dask-backed
    assert ds_lazy.temp.chunks is not None

    # Compute lazy result
    ds_lazy_computed = ds_lazy.compute()

    # Compare
    ds_eager.attrs.pop("history", None)
    ds_lazy_computed.attrs.pop("history", None)

    # Handle object strings
    def _sanitize(ds):
        for v in list(ds.data_vars) + list(ds.coords):
            if ds[v].dtype == object:
                ds[v] = ds[v].where(ds[v].notnull(), None)
        return ds

    ds_eager = _sanitize(ds_eager)
    ds_lazy_computed = _sanitize(ds_lazy_computed)

    xr.testing.assert_allclose(ds_eager, ds_lazy_computed)

    # Verify values (temp was 100 -> 10.0)
    assert ds_eager.temp.values[0, 0] == pytest.approx(10.0)


if __name__ == "__main__":
    pytest.main([__file__])
