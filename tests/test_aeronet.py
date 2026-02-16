import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from monetio import aeronet

DATA = Path(__file__).parent / "data"

try:
    import pytspack  # noqa: F401
except ImportError:
    has_pytspack = False
else:
    has_pytspack = True

# Decorator to skip tests that require external network access in CI
skip_on_ci = pytest.mark.skipif(
    os.environ.get("CI", "false").lower() == "true", reason="Skipped on CI"
)


@pytest.fixture
def mock_valid_sites():
    """Mock get_valid_sites to avoid network calls during tests."""
    with patch("monetio.readers.aeronet.get_valid_sites") as mock:
        mock.return_value = pd.DataFrame(
            {
                "siteid": ["Mauna_Loa", "SERC", "Cart_Site", "Chilbolton", "Banana_River"],
                "longitude": [-155.6, -76.5, -97.5, -1.4, -80.6],
                "latitude": [19.5, 38.9, 36.6, 51.1, 28.4],
                "elevation": [3397.0, 10.0, 315.0, 84.0, 2.0],
            }
        )
        yield mock


def test_build_url_required_param_checks(mock_valid_sites):
    # Default (nothing set; `dates`, `prod``, `daily` required)
    a = aeronet.AERONET()
    with pytest.raises(AssertionError):
        a.build_url()

    # Adding dates
    a.dates = pd.date_range("2021/08/01", "2021/08/03")
    with pytest.raises(AssertionError):
        a.build_url()

    # Adding prod
    a.prod = "AOD15"
    with pytest.raises(AssertionError):
        a.build_url()

    # Adding daily (now should work)
    a.daily = 20
    a.build_url()


def test_build_url_bad_prod(mock_valid_sites):
    dates = pd.date_range("2021/08/01", "2021/08/02")
    a = aeronet.AERONET()
    a.dates = dates
    a.daily = 10

    # Invalid non-inv product
    a.prod = "asdf"
    with pytest.raises(ValueError, match="invalid product"):
        a.build_url()

    # Good non-inv prod but inv_type set
    a.prod = "AOD15"
    a.inv_type = "ALM15"
    with pytest.raises(ValueError, match="invalid product"):
        a.build_url()

    # Bad inv_type
    a.inv_type = "asdf"
    with pytest.raises(ValueError, match="invalid inv type"):
        a.build_url()

    # Good inv type but prod isn't
    a.inv_type = "ALM15"
    with pytest.raises(ValueError, match="invalid product"):
        a.build_url()

    # Both good
    a.prod = "SIZ"
    a.build_url()


@skip_on_ci
def test_valid_sites_col_rename():
    assert (
        aeronet.get_valid_sites().columns == ["siteid", "longitude", "latitude", "elevation"]
    ).all()


@skip_on_ci
def test_add_data_bad_siteid():
    with pytest.raises(ValueError, match="invalid site"):
        aeronet.add_data(siteid="Rivendell")


@skip_on_ci
def test_add_data_one_site():
    dates = pd.date_range("2021/08/01", "2021/08/03")
    df = aeronet.add_data(dates, siteid="SERC", as_xarray=False)
    assert df.index.size > 0
    assert (df.siteid == "SERC").all()
    assert df.attrs["info"].startswith("AERONET Data Download")


@skip_on_ci
def test_add_data_inv():
    dates = pd.date_range("2021/08/01", "2021/08/02")

    df = aeronet.add_data(dates, inv_type="ALM15", product="SIZ", as_xarray=False)
    assert df.inversion_data_quality_level.eq("lev15").all()
    assert df.retrieval_measurement_scan_type.eq("Almucantar").all()

    df = aeronet.add_data(dates, inv_type="HYB15", product="SIZ")
    assert df.inversion_data_quality_level.eq("lev15").all()
    assert df.retrieval_measurement_scan_type.eq("Hybrid").all()


@skip_on_ci
@pytest.mark.parametrize("product", aeronet.AERONET._valid_prod_noninv)
def test_add_data_all_noninv(product):
    dates = pd.date_range("2021/08/01", "2021/08/02")
    site = "Mauna_Loa"

    df = aeronet.add_data(dates, product=product, siteid=site, as_xarray=False)
    assert df.index.size > 0


@skip_on_ci
def test_add_data_valid_empty_query():
    dates = pd.date_range("2021/08/01", "2021/08/02")
    site = "Banana_River"

    with pytest.raises(Exception, match="loading from URL .+ failed") as ei:
        aeronet.add_data(dates, product="AOD20", siteid=site)
    assert "valid query but no data found" in str(ei.value.__cause__)


def test_load_local():
    fp = DATA / "aeronet-AOD15-example.txt"
    assert fp.is_file()

    df = aeronet.add_local(fp, as_xarray=False)
    assert df.index.size > 0
    assert (df.siteid == "Mauna_Loa").all()
    assert df.attrs["info"].startswith("AERONET Data Download")


def test_load_local_inv():
    fp = DATA / "aeronet-inv-ALM15-SIZ-example.txt"
    assert fp.is_file()

    df = aeronet.add_local(fp, as_xarray=False)
    assert df.index.size > 0
    assert (df.siteid == "Cart_Site").all()


@skip_on_ci
def test_add_data_lunar():
    dates = pd.date_range("2021/08/01", "2021/08/02")
    df = aeronet.add_data(dates, lunar=True, daily=True)  # only daily-average data at this time
    assert df.index.size > 0

    dates = pd.date_range("2022/01/20", "2022/01/21")
    df = aeronet.add_data(dates, lunar=True, siteid="Chilbolton")
    assert df.index.size > 0


@skip_on_ci
def test_serial_freq():
    # For MM data proc example
    dates = pd.date_range(start="2019-09-01", end="2019-09-2", freq="h")
    df = aeronet.add_data(dates, freq="2h", n_procs=1, as_xarray=False)
    assert (
        pd.DatetimeIndex(sorted(df.time.unique()))
        == pd.date_range("2019-09-01", freq="2h", periods=12)
    ).all()


@skip_on_ci
@pytest.mark.skipif(has_pytspack, reason="has pytspack")
def test_interp_without_pytspack():
    # For MM data proc example
    dates = pd.date_range(start="2019-09-01", end="2019-09-2", freq="h")
    standard_wavelengths = np.array([0.34, 0.44, 0.55, 0.66, 0.86, 1.63, 11.1]) * 1000
    with pytest.raises(RuntimeError, match="You must install pytspack"):
        aeronet.add_data(dates, n_procs=1, interp_to_aod_values=standard_wavelengths)


@skip_on_ci
@pytest.mark.skipif(not has_pytspack, reason="no pytspack")
def test_interp_with_pytspack():
    # For MM data proc example
    dates = pd.date_range(start="2019-09-01", end="2019-09-2", freq="h")
    standard_wavelengths = np.array([0.34, 0.44, 0.55, 0.66, 0.86, 1.63, 11.1]) * 1000
    with pytest.warns(UserWarning, match="Renaming duplicate AOD columns"):
        df = aeronet.add_data(
            dates, n_procs=1, interp_to_aod_values=standard_wavelengths, as_xarray=False
        )

    # Check for the new columns
    assert {f"aod_{int(wl)}nm" for wl in standard_wavelengths}.issubset(df.columns)

    # Check for renamed duplicate columns
    assert {c for c in df if c.startswith("aod_") and c.endswith("nm_orig")} == {
        "aod_340nm_orig",
        "aod_440nm_orig",
    }


@skip_on_ci
@pytest.mark.skipif(not has_pytspack, reason="no pytspack")
def test_interp_daily_with_pytspack():
    dates = pd.date_range(start="2019-09-01", end="2019-09-2", freq="h")
    standard_wavelengths = np.array([0.55]) * 1000
    df = aeronet.add_data(
        dates, daily=True, n_procs=1, interp_to_aod_values=standard_wavelengths, as_xarray=False
    )

    assert {f"aod_{int(wl)}nm" for wl in standard_wavelengths}.issubset(df.columns)


@skip_on_ci
@pytest.mark.parametrize(
    "dates",
    [
        pd.to_datetime(["2019-09-01", "2019-09-02"]),
        pd.to_datetime(["2019-09-01", "2019-09-03"]),
        pd.to_datetime(["2019-09-01 00:00", "2019-09-01 12:00"]),
    ],
    ids=[
        "one day",
        "two days",
        "half day",
    ],
)
def test_issue100(dates, request):
    df1 = aeronet.add_data(dates, n_procs=1, as_xarray=False)
    df2 = aeronet.add_data(dates, n_procs=2, as_xarray=False)
    assert len(df1) == len(df2)
    if request.node.callspec.id == "two days":
        df1_ = df1.sort_values(["time", "siteid"]).reset_index(drop=True)
        df2_ = df2.sort_values(["time", "siteid"]).reset_index(drop=True)
        assert df1_.equals(df2_)
    else:
        assert df1.equals(df2)
    assert dates[0] < df1.time.min() < df1.time.max() < dates[-1]
