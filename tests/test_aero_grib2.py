import numpy as np
import xarray as xr

from monetio.readers.grib2 import Grib2Reader


def test_grib2_reader_open():
    ds = xr.Dataset(
        {"tmp": (("lat_0", "lon_0"), np.random.rand(10, 20))},
        coords={"lat_0": np.arange(10), "lon_0": np.arange(20)},
    )

    reader = Grib2Reader()

    import unittest.mock as mock

    with mock.patch("monetio.readers.drivers.XarrayDriver.open", return_value=ds):
        res = reader.open_dataset("dummy.grib2")

    assert "latitude" in res.coords
    assert "longitude" in res.coords
    assert "history" in res.attrs
    assert "grib2io" in res.attrs["history"]


def test_grib2_reader_harmonize():
    ds = xr.Dataset(
        {"tmp": (("lat", "lon"), np.random.rand(10, 20))},
        coords={
            "lat": np.arange(10),
            "lon": np.arange(20),
            "valid_time": [np.datetime64("2023-01-01")],
        },
    )

    reader = Grib2Reader()
    res = reader.harmonize(ds)

    assert "latitude" in res.coords
    assert "longitude" in res.coords
    # valid_time should be renamed to time
    assert "time" in res.variables
