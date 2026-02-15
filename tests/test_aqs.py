import os

import pandas as pd
import pytest

from monetio import aqs

# Skip on CI because epa.gov can be unreliable
skip_on_ci = pytest.mark.skipif(
    os.environ.get("CI", "false").lower() == "true", reason="Skipped on CI"
)


@skip_on_ci
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
            as_xarray=False,
        )
    assert (df.variable == "").sum() == 0
    t = df.time
    assert ((t.dt.year == 2019) & (t.dt.month == 8)).all()


@skip_on_ci
def test_aqs_daily_wide():
    dates = pd.date_range(start="2019-08-01", end="2019-08-31", freq="D")
    df = aqs.add_data(
        dates,
        param=["O3", "PM2.5"],
        network="IMPROVE",
        wide_fmt=True,
        daily=True,
        as_xarray=False,
    )
    t = df.time
    assert ((t.dt.year == 2019) & (t.dt.month == 8)).all()
