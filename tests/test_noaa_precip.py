import numpy as np
import pandas as pd
import pytest
import xarray as xr

import monetio


@pytest.mark.parametrize(
    ("source", "native", "standard"),
    [
        ("prism", "ppt", "precipitation_amount"),
        ("ccpa", "APCP_surface", "precipitation_amount"),
        ("nohrsc", "ASNOW", "snowfall_amount"),
    ],
)
def test_noaa_precip_local_netcdf_reader(tmp_path, source, native, standard):
    timestamp = np.datetime64("2024-01-02T00:00")
    ds = xr.Dataset(
        {native: (("time", "latitude", "longitude"), [[[1.25]]], {"units": "mm"})},
        coords={
            "time": [timestamp],
            "latitude": [40.0],
            "longitude": [-105.0],
        },
    )
    ds["time_bounds"] = (("time", "bounds"), [[timestamp, timestamp + np.timedelta64(6, "h")]])
    path = tmp_path / f"{source}.nc"
    ds.to_netcdf(path)

    result = monetio.load(source, files=str(path))
    assert standard in result
    assert result[standard].attrs["units"] == "mm"
    assert result[standard].item() == 1.25
    assert result.latitude.item() == 40.0
    assert result.longitude.item() == -105.0
    assert pd.Timestamp(result.time.item()) == pd.Timestamp(timestamp)
    assert "time_bounds" in result


def test_prism_native_bil_with_sidecars(tmp_path):
    from pyproj import CRS

    path = tmp_path / "PRISM_ppt_stable_4kmD1_20240102_bil.bil"
    np.array([[123, -9999], [300, 400]], dtype="<i2").tofile(path)
    path.with_suffix(".hdr").write_text(
        "NCOLS 2\nNROWS 2\nNBANDS 1\nNBITS 16\nBYTEORDER I\n"
        "ULXMAP 0.5\nULYMAP 1.5\nXDIM 1\nYDIM 1\nNODATA -9999\n"
    )
    path.with_suffix(".prj").write_text(CRS.from_epsg(4326).to_wkt())

    ds = monetio.load("prism", files=str(path))
    assert ds.precipitation_amount.dims == ("time", "y", "x")
    np.testing.assert_allclose(ds.precipitation_amount.values[0, 0, 0], 1.23)
    assert np.isnan(ds.precipitation_amount.values[0, 0, 1])
    assert ds.time.values[0] == np.datetime64("2024-01-02")
    assert ds.latitude.dims == ("y", "x")
    assert ds.longitude.dims == ("y", "x")
