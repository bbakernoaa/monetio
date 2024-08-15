import warnings

import pandas as pd
import pytest

from monetio.sat.nesdis_eps_viirs_aod_nrt import open_dataset, open_mfdataset

NOW = pd.Timestamp.now("UTC")
TODAY = NOW.floor("D")

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Converting to Period representation will drop timezone information.",
    )
    THIS_MONTH = TODAY.to_period("M").to_timestamp()

LAST_MONTH = THIS_MONTH - pd.DateOffset(months=1)
LAST_LAST_MONTH = LAST_MONTH - pd.DateOffset(months=1)


@pytest.mark.parametrize("res", [0.25, 0.1])
@pytest.mark.parametrize("sat", ["NOAA20", "SNPP"])
def test_open_dataset_daily(sat, res):
    # Note: only NRT
    date = (TODAY - pd.Timedelta(days=2)).tz_localize(None)
    ds = open_dataset(date, satellite=sat, data_resolution=res)

    assert date.strftime(r"%Y%m%d") in ds.attrs["dataset_name"]
    assert ds.attrs["spatial_resolution"].strip() == f"{res:.2f} degree"
    assert ds.attrs["satellite_name"] == ("Suomi NPP" if sat == "SNPP" else "NOAA 20")

    assert set(ds.dims) == {"time", "lat", "lon"}
    assert ds.sizes["time"] == 1
    assert ds.sizes["lat"] == int(180 / res)
    assert ds.sizes["lon"] == int(360 / res)
    assert (ds.time == pd.DatetimeIndex([date])).all()
    assert "AOD550" in ds.data_vars


@pytest.mark.parametrize("sat", ["NOAA20", "SNPP"])
def test_open_dataset_monthly(sat):
    # Seems like only one is stored
    if NOW - THIS_MONTH.tz_localize("UTC") > pd.Timedelta(hours=12):
        date = LAST_MONTH
    else:
        date = LAST_LAST_MONTH

    ds = open_dataset(date, satellite=sat, daily=False, data_resolution=0.25)
    assert ds.sizes["time"] == 1


def test_open_mfdataset():
    today = TODAY.tz_localize(None)
    dates = [today - pd.Timedelta(days=2), today - pd.Timedelta(days=3)]
    ds = open_mfdataset(dates)
    assert ds.sizes["time"] == len(dates)


def test_missing_date():
    from requests.exceptions import HTTPError

    with pytest.raises(HTTPError):
        open_dataset("1900-01-01")


def test_missing_date_mf():
    # No dsets collected
    with pytest.raises(ValueError, match="Files not available for product and dates"), pytest.warns(
        UserWarning, match="Failed to access file"
    ):
        open_mfdataset("1900-01-01")

    # Error during dsets collection
    with pytest.raises(RuntimeError, match="Failed to access file"):
        open_mfdataset("1900-01-01", error_missing=True)

    one_good = ["1900-01-01", TODAY.tz_localize(None) - pd.Timedelta(days=2)]
    with pytest.warns(UserWarning, match="Failed to access file"):
        ds = open_mfdataset(one_good)
        assert ds.sizes["time"] == 1

    with pytest.raises(RuntimeError, match="Failed to access file"), pytest.warns(
        UserWarning, match="Failed to access file"
    ):
        open_mfdataset(one_good, error_missing=True)
