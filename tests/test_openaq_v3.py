import os

import pandas as pd
import pytest

import monetio.obs.openaq_v3 as openaq

# TODO: check no camel case cols


if (
    os.environ.get("CI", "false").lower() not in {"false", "0"}
    and os.environ.get("OPENAQ_API_KEY", "") == ""
):
    # PRs from forks don't get the secret
    pytest.skip("no API key", allow_module_level=True)

LATLON_NCWCP = 38.9721, -76.9248
SITES_NEAR_NCWCP = [
    # AirGradient monitor
    1236068,
    1719392,
    # # PurpleAir sensors
    # 1118827,
    # 357301,
    # 273440,
    # 271155,
    # NASA GSFC
    2978434,
    # Beltsville (AirNow)
    3832,
    843,
]


def test_get_parameters():
    params = openaq.get_parameters()
    assert 20 <= len(params) <= 100
    assert params.id.nunique() == len(params)
    assert params.name.nunique() < len(params), "dupes for different units etc."
    assert "pm25" in params.name.values
    assert "o3" in params.name.values


def test_get_locations():
    sites = openaq.get_locations(npages=2, limit=100)
    assert len(sites) <= 200
    assert sites.siteid.nunique() == len(sites)
    assert sites.dtypes["first_time"] == "datetime64[ns]"
    assert sites.dtypes["last_time"] == "datetime64[ns]"
    assert sites.dtypes["latitude"] == "float64"
    assert sites.dtypes["longitude"] == "float64"
    assert sites["latitude"].isnull().sum() == 0
    assert sites["longitude"].isnull().sum() == 0

    # Check that we didn't end up with unexpected non-scalar columns
    for col in sites.columns:
        is_scalar = sites[col].apply(lambda x: pd.api.types.is_scalar(x)).all()
        assert is_scalar or col in {"parameters", "sensor_ids"}

    # Check that pm25 and o3 are in the unique parameters
    unique_params = set(sites["parameters"].sum())
    assert {"pm25", "o3"} <= unique_params
