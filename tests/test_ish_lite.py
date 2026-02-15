import pandas as pd
import pytest

from monetio import ish_lite

try:
    import requests

    requests.head("https://www1.ncdc.noaa.gov/pub/data/noaa/")
except Exception:
    pytest.skip("NCEI server issues", allow_module_level=True)


def test_ish_read_history():
    dates = pd.date_range("2020-09-01", "2020-09-02")
    ish = ish_lite.ISH()
    ish.dates = dates
    ish.read_ish_history()

    df = ish.history

    assert len(df) > 0
    assert {"latitude", "longitude", "begin", "end"} < set(df.columns)
    for col in ["begin", "end"]:
        assert pd.api.types.is_datetime64_any_dtype(df[col])
        assert (df[col].dt.hour == 0).all()

    assert df.station_id.nunique() == len(df), "unique ID for station"


def test_ish_lite_one_site():
    dates = pd.date_range("2020-09-01", "2020-09-02")
    site = "72224400358"  # "College Park AP"

    df = ish_lite.add_data(dates, site=site, as_xarray=False)

    assert (df.siteid == site).all(), "correct site"
    assert (df.time.diff().dropna() == pd.Timedelta("1h")).all(), "hourly data"
    assert len(df) == 24, "resampled from sub-hourly, so no hour 0 on second day"

    assert {
        "usaf",
        "wban",
        "latitude",
        "longitude",
        "country",
        "state",
    } < set(df.columns), "useful site metadata"
    assert {
        "time",
        "temp",
        "dew_pt_temp",
        "press",
        "wdir",
        "ws",
        "sky_condition",
        "precip_1hr",
        "precip_6hr",
    } < set(df.columns), "data columns"
    assert (df.temp < 100).all(), "temp in degC"
    assert (df.dew_pt_temp < 100).all(), "temp in degC"


@pytest.mark.parametrize("resample", [False, True])
def test_ish_lite_one_site_empty(resample):
    dates = pd.date_range("2020-09-01", "2020-09-02")
    site = "99816999999"  # "Delaware Reserve"

    df = ish_lite.add_data(dates, site=site, resample=resample, as_xarray=False)
    assert df.empty


def test_ish_lite_resample():
    dates = pd.date_range("2020-09-01", "2020-09-02")
    site = "72224400358"  # "College Park AP"
    freq = "3h"

    df = ish_lite.add_data(dates, site=site, resample=True, window=freq, as_xarray=False)

    assert (df.time.diff().dropna() == pd.Timedelta(freq)).all()
    assert len(df) == 8
