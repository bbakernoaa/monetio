import numpy as np
import pytest
import xarray as xr

from monetio.readers.nesdis_viirs_jrr import VIIRSJRRReader, viirs_jrr_preprocess


@pytest.mark.parametrize("product", ["AOD", "ADP"])
def test_viirs_jrr_preprocess(product):
    # Create dummy dataset
    ds = xr.Dataset(
        {
            "Latitude": (("Rows", "Columns"), np.zeros((3, 4))),
            "Longitude": (("Rows", "Columns"), np.zeros((3, 4))),
        },
        attrs={"time_coverage_start": "2024-01-01T00:00:00Z"},
    )

    if product == "AOD":
        ds["AOD550"] = (("Rows", "Columns"), np.ones((3, 4)))
    elif product == "ADP":
        ds["Smoke"] = (("Rows", "Columns"), np.ones((3, 4)))

    ds_out = viirs_jrr_preprocess(ds, product=product)

    assert "latitude" in ds_out.coords
    assert "longitude" in ds_out.coords
    assert "time" in ds_out.coords
    assert ds_out.y.size == 3
    assert ds_out.x.size == 4

    if product == "AOD":
        assert "aod_550" in ds_out.data_vars
    elif product == "ADP":
        assert "Smoke" in ds_out.data_vars


def test_viirs_jrr_build_urls(monkeypatch):
    class MockS3:
        def glob(self, pattern):
            if "VIIRS-JRR-AOD" in pattern:
                return ["bucket/VIIRS-JRR-AOD/2024/01/01/file1.nc"]
            if "VIIRS-JRR-ADP" in pattern:
                return ["bucket/VIIRS-JRR-ADP/2024/01/01/file2.nc"]
            return []

    monkeypatch.setattr("s3fs.S3FileSystem", lambda **kwargs: MockS3())

    reader = VIIRSJRRReader()
    urls_aod = reader.build_urls("2024-01-01", satellite="snpp", product="AOD")
    urls_adp = reader.build_urls("2024-01-01", satellite="snpp", product="ADP")

    assert len(urls_aod) == 1
    assert "file1.nc" in urls_aod[0]
    assert len(urls_adp) == 1
    assert "file2.nc" in urls_adp[0]


def test_viirs_jrr_stitching(monkeypatch):
    class MockDriver:
        def open(self, files, **kwargs):
            return xr.Dataset({"aod": (("time", "y", "x"), np.ones((len(files), 3, 4)))})

    reader = VIIRSJRRReader()
    monkeypatch.setattr(reader, "driver", MockDriver())

    ds = reader.open_dataset(files=["f1.nc", "f2.nc"], product="AOD")
    assert ds.aod.sizes["time"] == 2
