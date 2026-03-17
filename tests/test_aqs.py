import pandas as pd
import pytest

import functools

import requests

from monetio import aqs


def wrap_network_test(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (requests.exceptions.RequestException, RuntimeError, ValueError) as e:
            if isinstance(e, AssertionError):
                raise
            pytest.skip(f"Network or data retrieval error: {e}")

    return wrapper


@wrap_network_test
def test_aqs_daily_long():
    # For MM data proc example
    dates = pd.date_range(start="2019-08-01", end="2019-08-31", freq="D")
    # Note: will retrieve full year
    network = "NCORE"  # CSN NCORE CASTNET
    with pytest.warns(UserWarning, match="Short names not available for these variables"):
        df = aqs.add_data(
            dates,
            param=["PM10SPEC"],
            network=network,
            wide_fmt=False,
            daily=True,
        )
    assert (df.variable == "").sum() == 0
    t = df.time
    assert ((t.dt.year == 2019) & (t.dt.month == 8)).all()


@wrap_network_test
def test_aqs_daily_wide():
    dates = pd.date_range(start="2019-08-01", end="2019-08-31", freq="D")
    df = aqs.add_data(
        dates,
        param=["O3", "PM2.5"],
        network="IMPROVE",
        wide_fmt=True,
        daily=True,
    )
    t = df.time
    assert ((t.dt.year == 2019) & (t.dt.month == 8)).all()
