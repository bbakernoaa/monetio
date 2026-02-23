import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monetio.readers import ish_lite


@pytest.fixture
def mock_ish_lite_file(tmp_path):
    """Create a mock ISH Lite file."""
    # Columns: year, month, day, hour, temp, dew_pt_temp, press, wdir, ws, sky, precip1, precip6
    # Note: data is multiplied by 10 in the file for some columns
    data = []
    for h in range(24):
        data.append([2020, 9, 1, h, 200 + h, 150, 10130, 180, 50, 0, 0, 0])

    df = pd.DataFrame(data)
    # 722244-00358-2020.gz
    fname = tmp_path / "722244-00358-2020.gz"
    df.to_csv(fname, sep=" ", index=False, header=False, compression="gzip")
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


def test_read_ish_lite_file(mock_ish_lite_file):
    df = ish_lite.read_ish_lite_file(mock_ish_lite_file)
    assert len(df) == 24
    assert "temp" in df.columns
    assert df.temp.iloc[0] == 20.0
    assert df.siteid.iloc[0] == "72224400358"


def test_ish_lite_reader_eager(mock_ish_lite_file, mock_history_file, monkeypatch):
    # Mock history file URL and ISH class behavior
    def mock_init(self):
        self.history_file = mock_history_file
        self.history = None
        self.dates = None
        self.verbose = False
        self.source = "local"

    monkeypatch.setattr(ish_lite.ISH, "__init__", mock_init)

    dates = pd.date_range("2020-09-01", "2020-09-02")
    # Mock build_urls to return our mock file
    monkeypatch.setattr(
        ish_lite.ISH,
        "build_urls",
        lambda self, dates, sites, lite=False: pd.DataFrame({"name": [mock_ish_lite_file]}),
    )

    # Test as_xarray=False
    df = ish_lite.add_data(dates, site="72224400358", as_xarray=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 24
    assert "latitude" in df.columns
    assert df.latitude.iloc[0] == 33.0

    # Test as_xarray=True
    ds = ish_lite.add_data(dates, site="72224400358", as_xarray=True)
    assert isinstance(ds, xr.Dataset)
    assert "temp" in ds.data_vars
    assert ds.sizes["time"] == 24


def test_ish_lite_reader_lazy(mock_ish_lite_file, mock_history_file, monkeypatch):
    pytest.importorskip("dask")
    import dask.dataframe as dd

    def mock_init(self):
        self.history_file = mock_history_file
        self.history = None
        self.dates = None
        self.verbose = False
        self.source = "local"

    monkeypatch.setattr(ish_lite.ISH, "__init__", mock_init)
    monkeypatch.setattr(
        ish_lite.ISH,
        "build_urls",
        lambda self, dates, sites, lite=False: pd.DataFrame({"name": [mock_ish_lite_file]}),
    )

    dates = pd.date_range("2020-09-01", "2020-09-02")

    # Test lazy=True, as_xarray=False
    df = ish_lite.add_data(dates, site="72224400358", as_xarray=False, lazy=True)
    assert isinstance(df, dd.DataFrame)

    # Test lazy=True, as_xarray=True
    ds = ish_lite.add_data(dates, site="72224400358", as_xarray=True, lazy=True)
    assert isinstance(ds, xr.Dataset)
    # Check if data is dask-backed
    assert ds.temp.chunks is not None

    # Compute and verify
    ds_c = ds.compute()
    assert ds_c.sizes["time"] == 24
    # It might be 2D (time, node)
    if ds_c.temp.ndim == 2:
        assert np.allclose(ds_c.temp.values[0:3, 0], [20.0, 20.1, 20.2])
    else:
        assert np.allclose(ds_c.temp.values[0:3], [20.0, 20.1, 20.2])


def test_ish_lite_resample_eager(mock_ish_lite_file, mock_history_file, monkeypatch):
    def mock_init(self):
        self.history_file = mock_history_file
        self.history = None
        self.dates = None
        self.verbose = False
        self.source = "local"

    monkeypatch.setattr(ish_lite.ISH, "__init__", mock_init)
    monkeypatch.setattr(
        ish_lite.ISH,
        "build_urls",
        lambda self, dates, sites, lite=False: pd.DataFrame({"name": [mock_ish_lite_file]}),
    )

    dates = pd.date_range("2020-09-01", "2020-09-02")

    # Resample to 3h
    ds = ish_lite.add_data(dates, site="72224400358", as_xarray=True, resample=True, window="3h")
    assert ds.sizes["time"] == 8
    # Mean of [20.0, 20.1, 20.2] is 20.1
    assert np.isclose(ds.temp.values[0], 20.1)


def test_ish_lite_resample_lazy(mock_ish_lite_file, mock_history_file, monkeypatch):
    pytest.importorskip("dask")

    def mock_init(self):
        self.history_file = mock_history_file
        self.history = None
        self.dates = None
        self.verbose = False
        self.source = "local"

    monkeypatch.setattr(ish_lite.ISH, "__init__", mock_init)
    monkeypatch.setattr(
        ish_lite.ISH,
        "build_urls",
        lambda self, dates, sites, lite=False: pd.DataFrame({"name": [mock_ish_lite_file]}),
    )

    dates = pd.date_range("2020-09-01", "2020-09-02")

    # Resample to 3h (lazy)
    ds = ish_lite.add_data(
        dates, site="72224400358", as_xarray=True, lazy=True, resample=True, window="3h"
    )
    assert ds.temp.chunks is not None
    ds_c = ds.compute()
    assert ds_c.sizes["time"] == 8
    assert np.isclose(ds_c.temp.values[0], 20.1)
