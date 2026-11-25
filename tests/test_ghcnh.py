"""GHCNh reader tests: URL building, selection, QC masking, tidy conversion.

All tests use synthetic station-year files; no network access.
"""

import numpy as np
import pandas as pd
import pytest

from monetio.readers.ghcnh import (
    GHCNhReader,
    build_urls,
    read_ghcnh_file,
    select_stations,
)


def _stations():
    return pd.DataFrame(
        {
            "siteid": ["USW00094728", "USW00014739", "ACL000BARA9"],
            "latitude": [40.7789, 42.3606, 17.591],
            "longitude": [-73.9692, -71.0097, -61.821],
            "elevation": [42.7, 6.0, 5.0],
            "state": ["NY", "MA", "TX"],
            "country": ["US", "US", "AG"],
        }
    )


def _write_psv(path, flagged=False):
    """Write a minimal GHCNh-style PSV station-year file."""
    qc = "X" if flagged else ""
    lines = [
        "STATION|DATE|LATITUDE|LONGITUDE|ELEVATION|temperature|temperature_Quality_Code|wind_speed|wind_speed_Quality_Code",
        f"USW00094728|2026-05-01T00:51:00|40.7789|-73.9692|42.7|15.0|{qc}|3.1|",
        "USW00094728|2026-05-01T01:51:00|40.7789|-73.9692|42.7|15.6||3.4|",
        "USW00094728|2026-05-02T00:51:00|40.7789|-73.9692|42.7|14.2||2.9|",
    ]
    path.write_text("\n".join(lines) + "\n")


def test_build_urls_per_station_year():
    dates = pd.date_range("2026-05-01", "2026-05-07")
    urls = build_urls(dates, _stations().iloc[:2])
    assert urls == [
        "s3://noaa-ghcnh-pds/hourly/access/by-year/2026/parquet/GHCNh_USW00094728_2026.parquet",
        "s3://noaa-ghcnh-pds/hourly/access/by-year/2026/parquet/GHCNh_USW00014739_2026.parquet",
    ]
    psv = build_urls(dates, ["USW00094728"], layout="psv")
    assert psv[0].endswith("/psv/GHCNh_USW00094728_2026.psv")


def test_build_urls_multi_year():
    dates = pd.DatetimeIndex(["2025-12-31", "2026-01-01"])
    urls = build_urls(dates, ["USW00094728"])
    assert len(urls) == 2
    assert "2025" in urls[0]
    assert "2026" in urls[1]


def test_build_urls_rejects_bad_layout():
    with pytest.raises(ValueError, match="layout"):
        build_urls(pd.date_range("2026-05-01", periods=1), ["X"], layout="csv")


def test_select_stations_requires_selection():
    with pytest.raises(ValueError, match="at least one"):
        select_stations(_stations())


def test_select_stations_box_order_latlon():
    out = select_stations(_stations(), box=[36.0, -80.0, 43.0, -70.0])
    assert set(out.siteid) == {"USW00094728", "USW00014739"}


def test_select_stations_country_and_ids():
    assert select_stations(_stations(), country="us").siteid.tolist() == [
        "USW00094728",
        "USW00014739",
    ]
    assert select_stations(_stations(), siteid="ACL000BARA9").siteid.tolist() == ["ACL000BARA9"]


def test_read_psv_tidy_and_units(tmp_path):
    path = tmp_path / "GHCNh_USW00094728_2026.psv"
    _write_psv(path)
    df = read_ghcnh_file(str(path), variables=["temperature", "wind_speed"])
    assert list(df.columns) == [
        "siteid",
        "time",
        "latitude",
        "longitude",
        "elevation",
        "temperature",
        "wind_speed",
    ]
    assert df["time"].dtype.kind == "M"
    np.testing.assert_allclose(df["temperature"].values, [15.0, 15.6, 14.2])
    assert "temperature_Quality_Code" not in df


def test_read_psv_quality_filter_masks_flagged(tmp_path):
    path = tmp_path / "GHCNh_USW00094728_2026.psv"
    _write_psv(path, flagged=True)
    df = read_ghcnh_file(str(path), variables=["temperature", "wind_speed"], quality_filter=True)
    assert np.isnan(df["temperature"].iloc[0])  # flagged record masked
    assert df["temperature"].iloc[1] == 15.6  # unflagged kept
    keep = read_ghcnh_file(str(path), variables=["temperature"], quality_filter=False)
    assert keep["temperature"].iloc[0] == 15.0  # filter off keeps value


def test_open_dataset_local_files_xarray(tmp_path, monkeypatch):
    path = tmp_path / "GHCNh_USW00094728_2026.psv"
    _write_psv(path)
    ds = GHCNhReader().open_dataset(files=str(path), variables=["temperature", "wind_speed"])
    assert "temperature" in ds
    assert ds["temperature"].attrs.get("units") == "degC"
    assert ds.attrs.get("ghcnh_quality_filter") is True
    assert np.isfinite(ds["temperature"].values).any()


def test_open_dataset_requires_dates_or_files():
    with pytest.raises(ValueError, match="dates"):
        GHCNhReader().open_dataset()
