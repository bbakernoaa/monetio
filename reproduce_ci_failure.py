import json
import pandas as pd
import xarray as xr
import os
from monetio.readers.pams import PAMSReader

# Replicate mock_pams_file fixture
def create_mock_pams(tmp_path):
    f = os.path.join(tmp_path, "pams_test.json")
    data = {
        "Data": [
            {
                "state_code": "24",
                "county_code": "001",
                "site_number": "0001",
                "date_gmt": "2023-01-01",
                "time_gmt": "00:00",
                "sample_measurement": 10.0,
                "units_of_measure": "Parts per billion",
                "latitude": 39.0,
                "longitude": -76.5,
            }
        ]
    }
    with open(f, "w") as out:
        json.dump(data, out)
    return f

tmp_dir = "tmp_reproduce"
os.makedirs(tmp_dir, exist_ok=True)
mock_file = create_mock_pams(tmp_dir)

print("Running reproduction of test_pams_reader_eager...")
reader = PAMSReader()
ds = reader.open_dataset(files=mock_file, as_xarray=True, lazy=False)
print("Dataset variables:", list(ds.data_vars))
if "obs" in ds.data_vars:
    print("obs attrs:", ds.obs.attrs)
    try:
        print("Units:", ds.obs.attrs["units"])
    except KeyError:
        print("KeyError: 'units' not found in obs.attrs")

print("\nRunning reproduction of test_pams_reader_lazy...")
ds_lazy = reader.open_dataset(files=mock_file, as_xarray=True, lazy=True)
print("Lazy Dataset variables:", list(ds_lazy.data_vars))
if "obs" in ds_lazy.data_vars:
    print("obs attrs (lazy):", ds_lazy.obs.attrs)
    try:
        print("Units (lazy):", ds_lazy.obs.attrs["units"])
    except KeyError:
        print("KeyError: 'units' not found in obs.attrs (lazy)")
