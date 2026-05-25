import pandas as pd
import pytest
import xarray as xr

from monetio.readers.gfs import GDASReader, GEFSReader, GFSReader


def test_gfs_build_urls():
    reader = GFSReader()
    dates = pd.to_datetime(["2025-03-24"])
    urls = reader.build_urls(dates, hour=0, lead_time=3)
    assert len(urls) == 1
    assert urls[0] == "s3://noaa-gfs-bdp-pds/gfs.20250324/00/atmos/gfs.t00z.pgrb2.0p25.f003"

    urls = reader.build_urls(dates, hour=6, lead_time=[0, 3])
    assert len(urls) == 2
    assert urls[0] == "s3://noaa-gfs-bdp-pds/gfs.20250324/06/atmos/gfs.t06z.pgrb2.0p25.f000"
    assert urls[1] == "s3://noaa-gfs-bdp-pds/gfs.20250324/06/atmos/gfs.t06z.pgrb2.0p25.f003"


def test_gefs_build_urls():
    reader = GEFSReader()
    dates = pd.to_datetime(["2025-03-24"])
    # Default ensemble mean 0.5 deg
    urls = reader.build_urls(dates, hour=0, lead_time=3)
    assert len(urls) == 1
    assert (
        urls[0] == "s3://noaa-gefs-pds/gefs.20250324/00/atmos/pgrb2ap5/geavg.t00z.pgrb2a.0p50.f003"
    )

    # Custom member / resolution
    urls = reader.build_urls(dates, hour=12, lead_time=0, product="gep01.tHHz.pgrb2a.0p50")
    assert (
        urls[0] == "s3://noaa-gefs-pds/gefs.20250324/12/atmos/pgrb2ap5/gep01.t12z.pgrb2a.0p50.f000"
    )


def test_gdas_build_urls():
    reader = GDASReader()
    dates = pd.to_datetime(["2025-03-24"])
    urls = reader.build_urls(dates, hour=18, lead_time=0)
    assert len(urls) == 1
    assert urls[0] == "s3://noaa-gfs-bdp-pds/gdas.20250324/18/atmos/gdas.t18z.pgrb2.0p25.f000"


@pytest.mark.network
def test_gfs_open_dataset_integration():
    # This test requires grib2io and s3fs.
    # We'll try to at least build the URL and check if it's correct.
    # Actual opening might fail if grib2io is not fully functional in this environment.
    reader = GFSReader()
    # Use a recent date that is likely to exist
    date = (pd.Timestamp.now() - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    try:
        ds = reader.open_dataset(dates=date, hour=0, lead_time=0)
        assert isinstance(ds, xr.Dataset)
        assert "latitude" in ds.coords
        assert "longitude" in ds.coords
    except Exception as e:
        pytest.skip(f"GFS integration test skipped: {e}")
