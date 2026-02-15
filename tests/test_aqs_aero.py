import numpy as np
import pandas as pd
import pytest

from monetio.readers.aqs import AQSReader


def create_mock_aqs_file(fn, daily=False):
    if daily:
        # Simplified daily format based on load_aqs_file
        data = {
            "Date Local": ["2023-01-01", "2023-01-01"],
            "State Code": ["01", "01"],
            "County Code": ["001", "001"],
            "Site Num": ["0001", "0002"],
            "Parameter Code": [44201, 88101],
            "POC": [1, 1],
            "Latitude": [34.0, 35.0],
            "Longitude": [-86.0, -87.0],
            "Datum": ["WGS84", "WGS84"],
            "Parameter Name": ["Ozone", "PM2.5 - Local Conditions"],
            "Sample Duration": ["1 HOUR", "1 HOUR"],
            "Pollutant Standard": ["Ozone 8-hour 2015", "PM25 24-hour 2012"],
            "Units of Measure": ["Parts per billion", "Micrograms/cubic meter (LC)"],
            "Event Type": ["None", "None"],
            "Observation Count": [24, 24],
            "Observation Percent": [100, 100],
            "Sample Measurement": [40.0, 12.0],
            "1st Max Value": [45.0, 15.0],
            "1st Max Hour": [14, 10],
            "AQI": [37, 50],
            "Method Code": ["047", "145"],
            "Method Name": ["Instrumental", "Gravimetric"],
            "Local Site Name": ["Site 1", "Site 2"],
            "Address": ["123 St", "456 St"],
            "State Name": ["Alabama", "Alabama"],
            "County Name": ["Autauga", "Autauga"],
            "City Name": ["Prattville", "Prattville"],
            "MSA Name": ["Montgomery, AL", "Montgomery, AL"],
            "Date of Last Change": ["2023-02-01", "2023-02-01"],
        }
        # Add dummy columns to match renameddcols length if needed
        # renameddcols has 29 columns
        df = pd.DataFrame(data)
        # Ensure it has exactly 29 columns for the logic in load_aqs_file
        # Currently it has 29.
    else:
        data = {
            "Date GMT": ["2023-01-01", "2023-01-01"],
            "Time GMT": ["12:00", "13:00"],
            "Date Local": ["2023-01-01", "2023-01-01"],
            "Time Local": ["06:00", "07:00"],
            "State Code": ["01", "01"],
            "County Code": ["001", "001"],
            "Site Num": ["0001", "0001"],
            "Parameter Code": [44201, 44201],
            "POC": [1, 1],
            "Latitude": [34.0, 34.0],
            "Longitude": [-86.0, -86.0],
            "Sample Measurement": [40.0, 42.0],
            "Units of Measure": ["Parts per billion", "Parts per billion"],
            "Parameter Name": ["Ozone", "Ozone"],
        }
        df = pd.DataFrame(data)

    df.to_csv(fn, index=False)


@pytest.fixture
def mock_aqs_hourly(tmp_path):
    fn = tmp_path / "hourly_44201_2023.csv"
    create_mock_aqs_file(fn, daily=False)
    return str(fn)


def test_aqs_eager_vs_lazy(mock_aqs_hourly):
    dates = pd.date_range(start="2023-01-01", end="2023-01-02", freq="h")
    reader = AQSReader()

    # Eager
    df_eager = reader.open_dataset(
        files=mock_aqs_hourly, dates=dates, as_xarray=False, lazy=False, wide_fmt=False
    )

    # Lazy
    df_lazy = reader.open_dataset(
        files=mock_aqs_hourly, dates=dates, as_xarray=False, lazy=True, wide_fmt=False
    )

    assert hasattr(df_lazy, "compute")
    pd.testing.assert_frame_equal(df_eager, df_lazy.compute())


def test_aqs_xarray_eager_vs_lazy(mock_aqs_hourly):
    dates = pd.date_range(start="2023-01-01", end="2023-01-02", freq="h")
    reader = AQSReader()

    # Eager
    ds_eager = reader.open_dataset(
        files=mock_aqs_hourly, dates=dates, as_xarray=True, lazy=False, wide_fmt=False
    )

    # Lazy
    ds_lazy = reader.open_dataset(
        files=mock_aqs_hourly, dates=dates, as_xarray=True, lazy=True, wide_fmt=False
    )

    # Check that lazy one is indeed lazy (Dask-backed)
    assert ds_lazy.obs.chunks is not None

    # Eager one is 2D (time, node), Lazy one is 1D (node,)
    # To compare, we can flatten the Eager one or check values.
    # We'll check that the flattened 'obs' values match.
    np.testing.assert_allclose(ds_eager.obs.values.flatten(), ds_lazy.obs.compute().values)

    # Check history
    assert "Read AQS data" in ds_eager.attrs["history"]
    assert "Converted to xarray Dataset" in ds_eager.attrs["history"]


def test_aqs_unit_conversion(tmp_path):
    fn = tmp_path / "hourly_units_2023.csv"
    data = {
        "Date GMT": ["2023-01-01", "2023-01-01"],
        "Time GMT": ["12:00", "13:00"],
        "State Code": ["01", "01"],
        "County Code": ["001", "001"],
        "Site Num": ["0001", "0001"],
        "Parameter Code": [61103, 62101],  # WS, TEMP
        "Sample Measurement": [10.0, 77.0],
        "Units of Measure": ["Knots", "Degrees Fahrenheit"],
        "Parameter Name": ["Wind Speed", "Temperature"],
    }
    pd.DataFrame(data).to_csv(fn, index=False)

    dates = pd.date_range(start="2023-01-01", end="2023-01-02", freq="h")
    reader = AQSReader()
    df = reader.open_dataset(
        files=str(fn), dates=dates, as_xarray=False, lazy=False, wide_fmt=False
    )

    # Check WS conversion: 10 Knots * 0.51444 = 5.1444 m/s
    ws = df.loc[df.variable == "WS", "obs"].values[0]
    assert np.isclose(ws, 5.1444)
    assert df.loc[df.variable == "WS", "units"].values[0] == "m/s"

    # Check Temp conversion: (77 F + 459.67) * 5/9 = 298.15 K
    temp = df.loc[df.variable == "TEMP", "obs"].values[0]
    assert np.isclose(temp, 298.15)
    assert df.loc[df.variable == "TEMP", "units"].values[0] == "k"
