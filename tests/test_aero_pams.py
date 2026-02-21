import json

import pytest
import xarray as xr

from monetio.readers.pams import PAMSReader


def test_pams_eager_lazy(tmp_path):
    # Create a dummy PAMS JSON file
    dummy_data = {
        "Data": [
            {
                "state_code": 1,
                "county_code": 1,
                "site_number": 1,
                "date_gmt": "2023-01-01",
                "time_gmt": "00:00",
                "date_local": "2023-01-01",
                "time_local": "01:00",
                "sample_measurement": 10.0,
                "units_of_measure": "Parts per billion",
                "latitude": 40.0,
                "longitude": -100.0,
                "parameter": "Ozone",
            }
        ]
    }

    f = tmp_path / "test.json"
    with open(f, "w") as out:
        json.dump(dummy_data, out)

    reader = PAMSReader()

    # Test Eager Mode
    ds_eager = reader.open_dataset(files=[str(f)], as_xarray=True, lazy=False)
    assert isinstance(ds_eager, xr.Dataset)
    assert "obs" in ds_eager.data_vars
    assert ds_eager.obs.values[0] == 10.0
    # siteid is renamed to node during 2D expansion, but preserved as siteid coord
    assert ds_eager.coords["siteid"].values[0] == "010010001"

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
