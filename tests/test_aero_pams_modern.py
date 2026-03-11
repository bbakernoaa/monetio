import json

import pytest
import xarray as xr

from monetio.readers.pams import PAMSReader


def create_mock_pams_json(path):
    """Creates a mock PAMS JSON file."""
    data = {
        "Data": [
            {
                "state_code": "06",
                "county_code": "037",
                "site_number": "1103",
                "date_gmt": "2023-01-01",
                "time_gmt": "00:00",
                "sample_measurement": 1.5,
                "units_of_measure": "Parts per billion",
                "latitude": 34.0,
                "longitude": -118.0,
            },
            {
                "state_code": "06",
                "county_code": "037",
                "site_number": "1103",
                "date_gmt": "2023-01-01",
                "time_gmt": "01:00",
                "sample_measurement": 2.0,
                "units_of_measure": "Parts per billion",
                "latitude": 34.0,
                "longitude": -118.0,
            },
        ]
    }
    with open(path, "w") as f:
        json.dump(data, f)


def test_pams_protocol_compliance(tmp_path):
    """Verify PAMS processing is backend-agnostic and lazy-friendly."""
    json_path = tmp_path / "test_pams.json"
    create_mock_pams_json(json_path)

    reader = PAMSReader()

    # Test Eager
    res_eager = reader.open_dataset(files=str(json_path), lazy=False)

    # Test Lazy
    res_lazy = reader.open_dataset(files=str(json_path), lazy=True)

    # Check consistency
    xr.testing.assert_allclose(res_eager, res_lazy.compute())

    # Check history
    assert "history" in res_eager.attrs
    assert "Read PAMS data." in res_eager.attrs["history"]

    # Check column renaming
    assert "obs" in res_eager.data_vars
    assert res_eager.obs.attrs["units"] == "ppb"


if __name__ == "__main__":
    pytest.main([__file__])
