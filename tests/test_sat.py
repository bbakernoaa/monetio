import numpy as np
import xarray as xr
from monetio.sat import goes


def test_add_goes_bands_default():
    # Create a sample dataset
    data = np.random.rand(1, 10, 10)
    blue = xr.DataArray(data, dims=["time", "y", "x"], name="blue")
    red = xr.DataArray(data, dims=["time", "y", "x"], name="red")
    veggie = xr.DataArray(data, dims=["time", "y", "x"], name="veggie")
    dset = xr.Dataset({"blue": blue, "red": red, "veggie": veggie})

    # Call the function
    dset_tci = goes.add_goes_bands(dset)

    # Check that the tci variable was added
    assert "tci" in dset_tci.variables

    # Check the shape of the tci variable
    assert dset_tci.tci.shape == (1, 10, 10, 3)

    # Check the green band calculation
    green = (
        0.45 * dset.red.values
        + 0.1 * dset.veggie.values
        + 0.45 * dset.blue.values
    )
    expected_tci = np.stack([dset.red.values, green, dset.blue.values], axis=-1)
    np.testing.assert_allclose(dset_tci.tci.values, expected_tci)


def test_add_goes_bands_custom_names():
    # Create a sample dataset with different band names
    data = np.random.rand(1, 10, 10)
    blue_band = xr.DataArray(data, dims=["time", "y", "x"], name="b1")
    red_band = xr.DataArray(data, dims=["time", "y", "x"], name="b2")
    veggie_band = xr.DataArray(data, dims=["time", "y", "x"], name="b3")
    dset = xr.Dataset({"b1": blue_band, "b2": red_band, "b3": veggie_band})

    # Call the function with custom band names
    dset_tci = goes.add_goes_bands(dset, blue_band="b1", red_band="b2", veggie_band="b3")

    # Check that the tci variable was added
    assert "tci" in dset_tci.variables

    # Check the shape of the tci variable
    assert dset_tci.tci.shape == (1, 10, 10, 3)

    # Check the green band calculation
    green = (
        0.45 * dset.b2.values
        + 0.1 * dset.b3.values
        + 0.45 * dset.b1.values
    )
    expected_tci = np.stack([dset.b2.values, green, dset.b1.values], axis=-1)
    np.testing.assert_allclose(dset_tci.tci.values, expected_tci)


def test_add_goes_bands_custom_dims():
    # Create a sample dataset with different dimension names
    data = np.random.rand(1, 10, 10)
    blue = xr.DataArray(data, dims=["t", "lat", "lon"], name="blue")
    red = xr.DataArray(data, dims=["t", "lat", "lon"], name="red")
    veggie = xr.DataArray(data, dims=["t", "lat", "lon"], name="veggie")
    dset = xr.Dataset({"blue": blue, "red": red, "veggie": veggie})

    # Call the function
    dset_tci = goes.add_goes_bands(dset)

    # Check that the tci variable was added
    assert "tci" in dset_tci.variables

    # Check the shape of the tci variable
    assert dset_tci.tci.shape == (1, 10, 10, 3)

    # Check the dimension names
    assert dset_tci.tci.dims == ("t", "lat", "lon", "rgb")
