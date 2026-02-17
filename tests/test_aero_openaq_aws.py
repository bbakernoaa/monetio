import pandas as pd
import pytest
import xarray as xr

from monetio.readers.openaq_aws import OpenAQAWSReader


def test_openaq_aws_eager_lazy(tmp_path):
    # Create a dummy OpenAQ AWS CSV file
    dummy_data = pd.DataFrame(
        {
            "location_id": ["2178", "2178"],
            "sensor_id": ["1", "2"],
            "location": ["Test Site", "Test Site"],
            "datetime": ["2022-05-03 00:00:00+00:00", "2022-05-03 01:00:00+00:00"],
            "lat": [40.0, 40.0],
            "lon": [-100.0, -100.0],
            "parameter": ["pm25", "pm25"],
            "units": ["µg/m³", "µg/m³"],
            "value": [10.0, 15.0],
        }
    )

    f = tmp_path / "test.csv"
    dummy_data.to_csv(f, index=False)

    reader = OpenAQAWSReader()

    # Test Eager Mode
    ds_eager = reader.open_dataset(files=[str(f)], as_xarray=True, lazy=False)
    assert isinstance(ds_eager, xr.Dataset)
    assert "value" in ds_eager.data_vars
    assert ds_eager.value.values[0] == 10.0
    # siteid should be in coords or node
    if "siteid" in ds_eager.coords:
        assert ds_eager.coords["siteid"].values[0] == "2178"
    else:
        assert ds_eager.coords["node"].values[0] == "2178"

    # Test Lazy Mode
    ds_lazy = reader.open_dataset(files=[str(f)], as_xarray=True, lazy=True)
    assert isinstance(ds_lazy, xr.Dataset)

    # Assert identical values
    xr.testing.assert_allclose(
        ds_eager.drop_vars("history", errors="ignore"),
        ds_lazy.compute().drop_vars("history", errors="ignore"),
    )


if __name__ == "__main__":
    pytest.main([__file__])
