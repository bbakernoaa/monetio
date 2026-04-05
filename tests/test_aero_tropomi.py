import numpy as np
import pytest
import xarray as xr
from monetio.readers.tropomi import TROPOMIReader

@pytest.fixture
def mock_tropomi_file(tmp_path):
    """Create a mock TROPOMI L2 file structure with recognizable dimensions."""
    import netCDF4 as nc

    fn = tmp_path / "S5P_OFFL_L2_NO2_TEST.nc"
    with nc.Dataset(fn, "w") as ds:
        # Create groups
        prod = ds.createGroup("PRODUCT")
        scanline = 10
        ground_pixel = 20
        layer = 34

        # Create dimensions
        prod.createDimension("scanline", scanline)
        prod.createDimension("ground_pixel", ground_pixel)
        prod.createDimension("layer", layer)
        prod.createDimension("time", scanline) # Per scanline time to avoid alignment error

        # Create variables
        lat = prod.createVariable("latitude", "f4", ("scanline", "ground_pixel"))
        lat[:] = np.random.rand(scanline, ground_pixel)

        lon = prod.createVariable("longitude", "f4", ("scanline", "ground_pixel"))
        lon[:] = np.random.rand(scanline, ground_pixel)

        # time in PRODUCT group is often just a single reference time or per-scanline
        # Here we make it per scanline to match the expected 'y' dimension later
        time_var = prod.createVariable("time", "i4", ("time",))
        time_var[:] = np.full(scanline, 1600000000)

        dt = prod.createVariable("delta_time", "i4", ("scanline",))
        dt[:] = np.zeros(scanline)

        qa = prod.createVariable("qa_value", "f4", ("scanline", "ground_pixel"))
        qa[:] = np.ones((scanline, ground_pixel))

        no2 = prod.createVariable("nitrogendioxide_tropospheric_column", "f4", ("scanline", "ground_pixel"))
        no2[:] = np.random.rand(scanline, ground_pixel)

        # SUPPORT_DATA/INPUT_DATA
        supp_path = "PRODUCT/SUPPORT_DATA/INPUT_DATA"
        supp = ds.createGroup(supp_path)

        ps = supp.createVariable("surface_pressure", "f4", ("scanline", "ground_pixel"))
        ps[:] = np.full((scanline, ground_pixel), 101325.0)

    return str(fn)

def test_tropomi_eager_lazy(mock_tropomi_file):
    """Verify TROPOMIReader works with both Eager and Lazy backends."""
    reader = TROPOMIReader()

    # 1. Eager
    ds_eager = reader.open_dataset(files=mock_tropomi_file, lazy=False, calculate_pressure=False)
    assert "latitude" in ds_eager.coords
    assert ds_eager.latitude.dims == ("y", "x")
    assert "time" in ds_eager.dims

    # 2. Lazy
    ds_lazy = reader.open_dataset(files=mock_tropomi_file, lazy=True, calculate_pressure=False)
    try:
        import dask.array as da
        assert isinstance(ds_lazy.nitrogendioxide_tropospheric_column.data, da.Array)
    except ImportError:
        pass

    # 3. Equality
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())

def test_tropomi_qa_flagging(mock_tropomi_file):
    """Verify QA threshold masking works."""
    reader = TROPOMIReader()
    # Mask everything below 2.0 (our mock has 1.0)
    ds = reader.open_dataset(files=mock_tropomi_file, qa_threshold=2.0, calculate_pressure=False)
    assert ds.nitrogendioxide_tropospheric_column.isnull().all()
