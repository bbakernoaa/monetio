import json

import pytest
import xarray as xr

from monetio.readers.openaq import OpenAQReader


def test_openaq_eager_lazy(tmp_path):
    # Create a dummy OpenAQ JSON line file
    dummy_data = {
        "location": "Test Site",
        "city": "Test City",
        "country": "TS",
        "date": {"utc": "2023-01-01T00:00:00Z", "local": "2023-01-01T01:00:00+01:00"},
        "parameter": "pm25",
        "value": 10.0,
        "unit": "µg/m³",
        "coordinates": {"latitude": 40.0, "longitude": -100.0},
        "averagingPeriod": {"value": 1, "unit": "hours"},
    }

    f = tmp_path / "test.json"
    with open(f, "w") as out:
        out.write(json.dumps(dummy_data) + "\n")

    reader = OpenAQReader()

    # We can't easily use build_urls without network, so we test the parsing and processing
    # by mocking the file discovery or passing files directly if possible.
    # OpenAQReader.open_dataset calls a.add_data which calls build_urls.
    # Let's mock build_urls.

    from unittest.mock import patch

    with patch("monetio.readers.openaq.build_urls") as mock_urls:
        mock_urls.return_value = [str(f)]

        # Eager Mode
        ds_eager = reader.open_dataset(
            dates="2023-01-01", wide_fmt=True, as_xarray=True, lazy=False, expand2d=False
        )
        assert isinstance(ds_eager, xr.Dataset)
        assert "pm25_ugm3" in ds_eager.data_vars
        assert ds_eager.pm25_ugm3.values[0] == 10.0
        assert ds_eager.coords["siteid"].values[0].startswith("TS_")

        # Lazy Mode
        # Note: wide_fmt=True will force compute in my implementation too, with a warning.
        ds_lazy = reader.open_dataset(
            dates="2023-01-01", wide_fmt=True, as_xarray=True, lazy=True, expand2d=False
        )
        assert isinstance(ds_lazy, xr.Dataset)

        # Assert identical values
        xr.testing.assert_allclose(
            ds_eager.drop_vars("history", errors="ignore"),
            ds_lazy.compute().drop_vars("history", errors="ignore"),
        )


if __name__ == "__main__":
    pytest.main([__file__])
