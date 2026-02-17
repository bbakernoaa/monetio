import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monetio.readers.openaq import OpenAQReader


def test_openaq_eager_lazy(tmp_path):
    # Create a dummy OpenAQ JSON line file
    dummy_data = [
        {
            "location": "Test Site 1",
            "city": "Test City",
            "country": "TS",
            "date": {"utc": "2023-01-01T00:00:00Z", "local": "2023-01-01T01:00:00+01:00"},
            "parameter": "pm25",
            "value": 10.0,
            "unit": "µg/m³",
            "coordinates": {"latitude": 40.0, "longitude": -100.0},
            "averagingPeriod": {"value": 1, "unit": "hours"},
            "sourceName": "Source A",
            "sourceType": "government",
            "mobile": False,
        },
        {
            "location": "Test Site 1",
            "city": "Test City",
            "country": "TS",
            "date": {"utc": "2023-01-01T00:00:00Z", "local": "2023-01-01T01:00:00+01:00"},
            "parameter": "o3",
            "value": 50.0,
            "unit": "µg/m³",
            "coordinates": {"latitude": 40.0, "longitude": -100.0},
            "averagingPeriod": {"value": 1, "unit": "hours"},
            "sourceName": "Source A",
            "sourceType": "government",
            "mobile": False,
        },
    ]

    f = tmp_path / "test.json"
    with open(f, "w") as out:
        for item in dummy_data:
            out.write(json.dumps(item) + "\n")

    reader = OpenAQReader()

    # Test Eager Mode (Wide format)
    ds_eager = reader.open_dataset(files=[str(f)], wide_fmt=True, as_xarray=True, lazy=False)
    assert isinstance(ds_eager, xr.Dataset)
    assert "pm25_ugm3" in ds_eager.data_vars
    assert "o3_ppm" in ds_eager.data_vars
    assert ds_eager.pm25_ugm3.values[0] == 10.0
    # O3 conversion: 50 / 1990
    assert np.isclose(ds_eager.o3_ppm.values[0], 50.0 / 1990.0)
    # siteid is renamed to node during 2D expansion
    assert ds_eager.coords["node"].values[0].startswith("TS_")

    # Test Lazy Mode (Wide format)
    ds_lazy = reader.open_dataset(files=[str(f)], wide_fmt=True, as_xarray=True, lazy=True)
    assert isinstance(ds_lazy, xr.Dataset)

    # Assert identical values
    xr.testing.assert_allclose(
        ds_eager.drop_vars("history", errors="ignore"),
        ds_lazy.compute().drop_vars("history", errors="ignore"),
    )


def test_openaq_long_format(tmp_path):
    dummy_data = {
        "location": "Test Site 1",
        "city": "Test City",
        "country": "TS",
        "date": {"utc": "2023-01-01T00:00:00Z", "local": "2023-01-01T01:00:00+01:00"},
        "parameter": "pm25",
        "value": 10.0,
        "unit": "µg/m³",
        "coordinates": {"latitude": 40.0, "longitude": -100.0},
    }

    f = tmp_path / "test.json"
    with open(f, "w") as out:
        out.write(json.dumps(dummy_data) + "\n")

    reader = OpenAQReader()

    # Test Eager Mode (Long format)
    df_eager = reader.open_dataset(files=[str(f)], wide_fmt=False, as_xarray=False, lazy=False)
    assert isinstance(df_eager, pd.DataFrame)
    assert "variable" in df_eager.columns
    assert df_eager.loc[0, "variable"] == "pm25"
    assert df_eager.loc[0, "obs"] == 10.0

    # Test Lazy Mode (Long format)
    df_lazy = reader.open_dataset(files=[str(f)], wide_fmt=False, as_xarray=False, lazy=True)
    assert hasattr(df_lazy, "compute")

    pd.testing.assert_frame_equal(df_eager, df_lazy.compute())


if __name__ == "__main__":
    pytest.main([__file__])
