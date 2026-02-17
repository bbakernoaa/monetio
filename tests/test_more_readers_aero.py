import json

import pytest
import xarray as xr

from monetio.readers.pams import PAMSReader


def test_pams_eager_lazy(tmp_path):
    # Create a dummy PAMS JSON file
    dummy_data = {
        "Data": [
            {
                "state_code": "01",
                "county_code": "001",
                "site_number": "0001",
                "date_gmt": "2023-01-01",
                "time_gmt": "00:00",
                "date_local": "2023-01-01",
                "time_local": "01:00",
                "sample_measurement": 1.5,
                "units_of_measure": "Parts per billion Carbon",
                "parameter": "Ethane",
                "latitude": 40.0,
                "longitude": -100.0,
            }
        ]
    }

    f = tmp_path / "test.json"
    with open(f, "w") as out:
        json.dump(dummy_data, out)

    reader = PAMSReader()

    # Eager Mode
    ds_eager = reader.open_dataset(files=str(f), as_xarray=True, lazy=False, expand2d=False)
    assert isinstance(ds_eager, xr.Dataset)
    assert "obs" in ds_eager.data_vars
    assert ds_eager.units.values[0] == "ppbC"
    assert ds_eager.siteid.values[0] == "010010001"

    # Lazy Mode
    ds_lazy = reader.open_dataset(files=str(f), as_xarray=True, lazy=True, expand2d=False)
    assert isinstance(ds_lazy, xr.Dataset)
    assert hasattr(ds_lazy.obs.data, "dask")

    # Assert identical values
    xr.testing.assert_allclose(
        ds_eager.drop_vars("history", errors="ignore"),
        ds_lazy.compute().drop_vars("history", errors="ignore"),
    )


def test_cems_eager_lazy(tmp_path):
    # Create a dummy CEMS CSV file
    # Columns: state, facility name, orispl, fac id, unit id, op date, op hour, op time, gload (mw), sload (1000lb/hr), so2 mass (lbs), ...
    # Our reader is quite flexible with column names
    csv_content = "State,Facility Name,ORISPL,Date,Hour,LATITUDE,LONGITUDE,SO2 Mass (lbs)\n"
    csv_content += "MD,Test Plant,1234,2023-01-01,0,40.0,-100.0,10.5\n"

    f = tmp_path / "test.csv"
    f.write_text(csv_content)

    from monetio.readers.cems import CEMSReader

    reader = CEMSReader()

    # Eager Mode
    ds_eager = reader.open_dataset(files=str(f), as_xarray=True, lazy=False, expand2d=False)
    assert isinstance(ds_eager, xr.Dataset)
    assert "so2_lbs" in ds_eager.data_vars
    assert ds_eager.siteid.values[0] == "1234"

    # Lazy Mode
    ds_lazy = reader.open_dataset(files=str(f), as_xarray=True, lazy=True, expand2d=False)
    assert isinstance(ds_lazy, xr.Dataset)
    assert hasattr(ds_lazy.so2_lbs.data, "dask")

    # Assert identical values
    xr.testing.assert_allclose(
        ds_eager.drop_vars("history", errors="ignore"),
        ds_lazy.compute().drop_vars("history", errors="ignore"),
    )


if __name__ == "__main__":
    pytest.main([__file__])
