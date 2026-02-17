import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monetio.readers import ish


@pytest.fixture
def mock_ish_file(tmp_path):
    """Create a mock ISH file."""

    def make_line(date, hour, temp):
        line = (
            f"{100:04d}"  # varlength
            f"{'72224400358':11s}"
            f"{date:08d}"
            f"{hour:04d}"
            f"{'4':1s}"
            f"{'+33000':6s}"
            f"{'-087000':7s}"
            f"{'FM-15':5s}"
            f"{'+0100':5s}"
            f"{'TEST ':5s}"
            f"{'4   ':4s}"
            f"{180:03d}"  # wdir
            f"{'1':1s}"
            f"{'1':1s}"
            f"{50:04d}"  # ws (will be / 10 = 5.0)
            f"{'1':1s}"
            f"{99999:05d}"
            f"{'1':1s}"
            f"{'1':1s}"
            f"{'1':1s}"
            f"{999999:06d}"
            f"{'1':1s}"
            f"{'1':1s}"
            f"{'1':1s}"
            f"{temp:05d}"  # t (will be / 10)
            f"{'1':1s}"
            f"{150:05d}"  # dpt
            f"{'1':1s}"
            f"{10130:05d}"  # p
            f"{'1':1s}"
        )
        return line

    lines = []
    for h in range(24):
        lines.append(make_line(20200901, h * 100, 200 + h))

    import gzip

    fname = tmp_path / "722244-00358-2020.gz"
    with gzip.open(fname, "wt") as f:
        f.write("\n".join(lines))
    return str(fname)


@pytest.fixture
def mock_history_file(tmp_path):
    """Create a mock history file."""
    data = {
        "USAF": ["722244"],
        "WBAN": ["00358"],
        "STATION NAME": ["TEST SITE"],
        "CTRY": ["US"],
        "STATE": ["AL"],
        "LAT": [33.0],
        "LON": [-87.0],
        "ELEV(M)": [100.0],
        "BEGIN": ["20200101"],
        "END": ["20201231"],
    }
    df = pd.DataFrame(data)
    fname = tmp_path / "isd-history.csv"
    df.to_csv(fname, index=False)
    return str(fname)


def test_ish_eager_vs_lazy(mock_ish_file, mock_history_file, monkeypatch):
    """
    Double-Check Test: Verify Eager (Pandas) and Lazy (Dask) results are identical.
    Following the Aero Protocol.
    """
    from monetio.readers.ish import ISH

    def mock_init(self):
        self.history_file = mock_history_file
        self.history = None
        self.dates = None
        self.verbose = False
        self.source = "aws"

    monkeypatch.setattr(ISH, "__init__", mock_init)

    def mock_read_history(self, dates=None):
        df = pd.read_csv(mock_history_file, dtype=str)
        df.columns = [i.lower() for i in df.columns]
        df.loc[:, "usaf"] = df.usaf.astype("str").str.zfill(6)
        df.loc[:, "wban"] = df.wban.astype("str").str.zfill(5)
        df["station_id"] = df.usaf + df.wban
        df.rename(columns={"lat": "latitude", "lon": "longitude"}, inplace=True)
        self.history = df

    monkeypatch.setattr(ISH, "read_ish_history", mock_read_history)

    monkeypatch.setattr(
        ISH,
        "build_urls",
        lambda self, dates=None, sites=None: pd.DataFrame({"name": [mock_ish_file]}),
    )

    dates = pd.date_range("2020-09-01", "2020-09-02")
    site = "72224400358"

    # 1. Eager Load
    ds_eager = ish.ISHReader().open_dataset(
        dates=dates, site=site, as_xarray=True, lazy=False, resample=False
    )

    # 2. Lazy Load
    ds_lazy = ish.ISHReader().open_dataset(
        dates=dates, site=site, as_xarray=True, lazy=True, resample=False
    )

    assert ds_lazy.t.chunks is not None
    ds_lazy_computed = ds_lazy.compute()

    # Sanitization
    def _sanitize(ds):
        ds = ds.drop_vars("history", errors="ignore")
        ds.attrs.pop("history", None)
        for v in list(ds.data_vars) + list(ds.coords):
            if ds[v].dtype == object:
                ds[v] = ds[v].where(ds[v].notnull(), None)
        return ds

    ds_eager_clean = _sanitize(ds_eager.copy())
    ds_lazy_clean = _sanitize(ds_lazy_computed.copy())

    xr.testing.assert_allclose(ds_eager_clean, ds_lazy_clean)
    # to_xarray returns (time, node) by default
    assert np.allclose(ds_eager.t.values[0:3, 0], [20.0, 20.1, 20.2])


def test_ish_lazy_resample(mock_ish_file, mock_history_file, monkeypatch):
    """Verify that lazy resampling works via Xarray."""
    from monetio.readers.ish import ISH

    def mock_init(self):
        self.history_file = mock_history_file
        self.history = None
        self.dates = None
        self.verbose = False
        self.source = "aws"

    monkeypatch.setattr(ISH, "__init__", mock_init)

    def mock_read_history(self, dates=None):
        df = pd.read_csv(mock_history_file, dtype=str)
        df.columns = [i.lower() for i in df.columns]
        df.loc[:, "usaf"] = df.usaf.astype("str").str.zfill(6)
        df.loc[:, "wban"] = df.wban.astype("str").str.zfill(5)
        df["station_id"] = df.usaf + df.wban
        df.rename(columns={"lat": "latitude", "lon": "longitude"}, inplace=True)
        self.history = df

    monkeypatch.setattr(ISH, "read_ish_history", mock_read_history)

    monkeypatch.setattr(
        ISH,
        "build_urls",
        lambda self, dates=None, sites=None: pd.DataFrame({"name": [mock_ish_file]}),
    )

    dates = pd.date_range("2020-09-01 00:00", "2020-09-01 23:59", freq="h")
    site = "72224400358"

    # Lazy load with resample=True
    ds_lazy = ish.ISHReader().open_dataset(
        dates=dates, site=site, as_xarray=True, lazy=True, resample=True, window="3h"
    )

    assert ds_lazy.t.chunks is not None
    ds_resampled = ds_lazy.compute()

    assert len(ds_resampled.time) == 8
    # Mean of [20.0, 20.1, 20.2] is 20.1
    assert np.isclose(ds_resampled.t.values[0, 0], 20.1)


if __name__ == "__main__":
    pytest.main([__file__])
