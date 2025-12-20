import warnings

import pandas as pd
import pytest

from monetio.readers import READER_REGISTRY

# Get the reader
viirs_aod_nrt_reader = READER_REGISTRY["nesdis_eps_viirs_aod_nrt"]()

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
    ds = viirs_aod_nrt_reader.open_dataset(date=date, satellite=sat, data_resolution=res)

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

    ds = viirs_aod_nrt_reader.open_dataset(date=date, satellite=sat, daily=False, data_resolution=0.25)
    assert ds.sizes["time"] == 1


def test_open_mfdataset():
    today = TODAY.tz_localize(None)
    dates = [today - pd.Timedelta(days=2), today - pd.Timedelta(days=3)]
    ds = viirs_aod_nrt_reader.open_dataset(dates=dates)
    assert ds.sizes["time"] == len(dates)


def test_missing_date():
    with pytest.raises(Exception):
        viirs_aod_nrt_reader.open_dataset(date="1900-01-01", error_missing=True)


def test_missing_date_mf():
    # No dsets collected
    with pytest.warns(UserWarning, match="Failed to access file"):
        ds = viirs_aod_nrt_reader.open_dataset(dates="1900-01-01")
        assert len(ds.data_vars) == 0

    # Error during dsets collection
    with pytest.raises(Exception):
        viirs_aod_nrt_reader.open_dataset(dates="1900-01-01", error_missing=True)

    one_good = ["1900-01-01", TODAY.tz_localize(None) - pd.Timedelta(days=2)]
    with pytest.warns(UserWarning, match="Failed to access file"):
        ds = viirs_aod_nrt_reader.open_dataset(one_good)
        assert ds.sizes["time"] == 1

    with pytest.raises(Exception):
        viirs_aod_nrt_reader.open_dataset(one_good, error_missing=True)
