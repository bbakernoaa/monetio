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


def assert_columns_all_snake_case(df):
    assert all(df.columns.str.fullmatch(r"[a-z_]+"))


def test_get_parameters():
    params = openaq.get_parameters()
    assert_columns_all_snake_case(params)
    assert 20 <= len(params) <= 100
    assert params.id.nunique() == len(params)
    assert params.name.nunique() < len(params), "dupes for different units etc."
    assert "pm25" in params.name.values
    assert "o3" in params.name.values


def test_get_locations():
    sites = openaq.get_locations()
    assert_columns_all_snake_case(sites)
    assert 10_000 <= len(sites) < 50_000
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
        assert is_scalar or col in {"parameters", "parameter_ids", "sensor_ids"}

    # Check that pm25 and o3 are in the unique parameters
    unique_params = set(sites["parameters"].sum())
    assert {"pm25", "o3"} <= unique_params


def test_get_sensors():
    df = openaq.get_sensors("3832")
    assert_columns_all_snake_case(df)
    assert df.parameter.tolist() == ["so2", "o3"]


def test_add_data_sensor_ids():
    df = openaq.get_sensors("2978434")
    sensor_ids = df["id"].tolist()
    assert len(sensor_ids) == 1
    df = openaq.add_data(
        ["2024-08-01", "2024-08-08"],
        query_time_split="1D",
        sensor_ids=sensor_ids,
        threads=2,
    )
    assert_columns_all_snake_case(df)
    assert len(df) > 0


def test_add_data_sensor_limit():
    df = openaq.add_data(
        ["2019-08-01", "2019-08-02"],
        query_time_split=None,
        sensor_limit=10,
        threads=2,
    )
    assert_columns_all_snake_case(df)
    assert len(df) > 0
    assert df.sensor_id.nunique() <= 10

    df_wide = openaq._to_wide_fmt(df)
    assert df.query("parameter == 'pm25'").value.mean() == df_wide.pm25_ugm3.mean()
