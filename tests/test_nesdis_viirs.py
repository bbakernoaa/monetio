import numpy as np
import pytest
import xarray as xr

from monetio.readers.nesdis_eps_viirs import NESDISEPSVIIRSReader, nesdis_eps_viirs_preprocess
from monetio.readers.nesdis_viirs_jrr import VIIRSJRRAODReader, viirs_jrr_preprocess


def test_eps_build_urls():
    reader = NESDISEPSVIIRSReader()
    urls = reader.build_urls("2023-01-01")
    assert len(urls) == 1
    assert "ftp://ftp.star.nesdis.noaa.gov" in urls[0]
    assert "20230101" in urls[0]


def test_jrr_build_urls(monkeypatch):
    # Mock s3fs to avoid network calls
    class MockFS:
        def glob(self, pattern):
            return ["noaa-nesdis-snpp-pds/VIIRS-JRR-AOD/2023/01/01/test_file.nc"]

    monkeypatch.setattr("s3fs.S3FileSystem", lambda **kwargs: MockFS())

    reader = VIIRSJRRAODReader()
    urls = reader.build_urls("2023-01-01")
    assert len(urls) == 1
    assert urls[0] == "s3://noaa-nesdis-snpp-pds/VIIRS-JRR-AOD/2023/01/01/test_file.nc"


@pytest.mark.parametrize("lazy", [True, False])
def test_eps_preprocess(lazy):
    # Create dummy dataset
    ds = xr.Dataset(
        {"aot_ip_out": (("nlat", "nlon"), np.random.rand(720, 1440).astype(np.float32))},
        attrs={"time_coverage_start": "2023-01-01T00:00:00Z"},
    )
    if lazy:
        ds = ds.chunk({"nlat": 360, "nlon": 720})

    ds_out = nesdis_eps_viirs_preprocess(ds)

    assert "latitude" in ds_out.coords
    assert "longitude" in ds_out.coords
    assert "time" in ds_out.coords
    assert ds_out.aod_550.dims == ("time", "y", "x")
    assert ds_out.latitude.shape == (720, 1440)
    assert ds_out.aod_550.attrs["units"] == "1"

    if lazy:
        assert ds_out.aod_550.chunks is not None


def test_jrr_daily_aggregation():
    # Simulate multiple granules for a single day
    ds1 = xr.Dataset(
        {"AOD550": (("Rows", "Columns"), np.full((10, 10), 0.5, dtype=np.float32))},
        coords={
            "Latitude": (("Rows", "Columns"), np.zeros((10, 10), dtype=np.float32)),
            "Longitude": (("Rows", "Columns"), np.zeros((10, 10), dtype=np.float32)),
        },
        attrs={"time_coverage_start": "2023-01-01T00:00:00Z"},
    )
    ds2 = xr.Dataset(
        {"AOD550": (("Rows", "Columns"), np.full((10, 10), 1.5, dtype=np.float32))},
        coords={
            "Latitude": (("Rows", "Columns"), np.zeros((10, 10), dtype=np.float32)),
            "Longitude": (("Rows", "Columns"), np.zeros((10, 10), dtype=np.float32)),
        },
        attrs={"time_coverage_start": "2023-01-01T12:00:00Z"},
    )

    # Preprocess both
    p1 = viirs_jrr_preprocess(ds1)
    p2 = viirs_jrr_preprocess(ds2)

    # Concatenate (simulating open_mfdataset behavior)
    ds_combined = xr.concat([p1, p2], dim="time")

    assert ds_combined.sizes["time"] == 2
    assert "aod_550" in ds_combined.data_vars

    # Global daily average
    # All pixels in granule 1 are 0.5, all in granule 2 are 1.5.
    # Total mean should be 1.0
    global_mean = ds_combined.aod_550.mean()
    assert float(global_mean) == pytest.approx(1.0)

    # Daily resample
    daily_mean = ds_combined.aod_550.resample(time="1D").mean()
    assert daily_mean.sizes["time"] == 1
    assert float(daily_mean.isel(time=0).mean()) == pytest.approx(1.0)


@pytest.mark.parametrize("lazy", [True, False])
def test_jrr_preprocess(lazy):
    # Create dummy dataset based on observed JRR structure
    ds = xr.Dataset(
        {"AOD550": (("Rows", "Columns"), np.random.rand(100, 200).astype(np.float32))},
        coords={
            "Latitude": (("Rows", "Columns"), np.random.rand(100, 200).astype(np.float32)),
            "Longitude": (("Rows", "Columns"), np.random.rand(100, 200).astype(np.float32)),
        },
        attrs={"time_coverage_start": "2023-01-01T12:00:00Z"},
    )
    if lazy:
        ds = ds.chunk({"Rows": 50, "Columns": 100})

    ds_out = viirs_jrr_preprocess(ds)

    assert "latitude" in ds_out.coords
    assert "longitude" in ds_out.coords
    assert "time" in ds_out.coords
    assert ds_out.aod_550.dims == ("time", "y", "x")
    assert ds_out.y.size == 100
    assert ds_out.x.size == 200

    if lazy:
        assert ds_out.aod_550.chunks is not None
