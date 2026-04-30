import numpy as np
import pytest
import xarray as xr

from monetio.readers.ncep_grib import NCEPGribReader


def mock_ncep_grib_dataset():
    """Create a mock NCEP GRIB dataset with 1D lat/lon."""
    nlat = 10
    nlon = 20
    ntime = 1

    ds = xr.Dataset(
        {
            "TMP_P0_L1_GLL0": (("lat_0", "lon_0"), np.random.rand(nlat, nlon).astype(np.float32)),
        }
    )
    ds.coords["lat_0"] = np.linspace(90, -90, nlat).astype(np.float32)
    ds.coords["lon_0"] = np.linspace(0, 359, nlon).astype(np.float32)

    # Add some attributes with whitespace to test hygiene
    ds.attrs["TITLE"] = "  Mock NCEP GRIB Data  "
    ds.TMP_P0_L1_GLL0.attrs["units"] = " K "

    return ds


def test_ncep_grib_eager_lazy(tmp_path):
    ds_mock = mock_ncep_grib_dataset()
    fname = tmp_path / "test_ncep_grib.nc"
    # We use netcdf4 to save as NCEP GRIB readers often expect netCDF-like structure
    # when coming from pynio or similar engines.
    ds_mock.to_netcdf(fname, engine="h5netcdf")

    reader = NCEPGribReader()

    # 1. Eager Mode (using h5netcdf engine for the mock file)
    ds_eager = reader.open_dataset(files=str(fname), lazy=False, engine="h5netcdf")

    assert isinstance(ds_eager, xr.Dataset)
    assert "latitude" in ds_eager.coords
    assert "longitude" in ds_eager.coords
    assert ds_eager.latitude.ndim == 2
    assert ds_eager.longitude.ndim == 2
    assert ds_eager.attrs["TITLE"] == "Mock NCEP GRIB Data"
    assert ds_eager.TMP_P0_L1_GLL0.attrs["units"] == "K"

    # 2. Lazy Mode
    ds_lazy = reader.open_dataset(files=str(fname), chunks={"lat_0": 5, "lon_0": 10}, engine="h5netcdf")

    assert isinstance(ds_lazy, xr.Dataset)
    # Check if data is dask-backed
    assert hasattr(ds_lazy.TMP_P0_L1_GLL0.data, "dask")
    assert hasattr(ds_lazy.latitude.data, "dask")
    assert hasattr(ds_lazy.longitude.data, "dask")

    # 3. Consistency Check
    # Drop history for comparison as it contains timestamps
    xr.testing.assert_allclose(
        ds_eager.drop_vars("history", errors="ignore"),
        ds_lazy.compute().drop_vars("history", errors="ignore"),
    )

    # Verify provenance
    assert "history" in ds_eager.attrs
    assert "Generated 2D latitude/longitude coordinates lazily" in ds_eager.attrs["history"]


def test_ncep_grib_variable_promotion(tmp_path):
    """Test that lat_0/lon_0 are promoted to coords if they are only variables."""
    nlat = 5
    nlon = 10
    ds = xr.Dataset(
        {
            "TMP": (("lat_0", "lon_0"), np.random.rand(nlat, nlon).astype(np.float32)),
            "lat_0": (("lat_0",), np.linspace(90, -90, nlat).astype(np.float32)),
            "lon_0": (("lon_0",), np.linspace(0, 350, nlon).astype(np.float32)),
        }
    )
    # Note: we don't set them as coords
    fname = tmp_path / "test_promotion.nc"
    ds.to_netcdf(fname, engine="h5netcdf")

    reader = NCEPGribReader()
    ds_out = reader.open_dataset(files=str(fname), engine="h5netcdf")

    assert "latitude" in ds_out.coords
    assert "longitude" in ds_out.coords
    assert ds_out.latitude.ndim == 2


if __name__ == "__main__":
    pytest.main([__file__])
