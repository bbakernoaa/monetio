import numpy as np
import pytest
import xarray as xr

from monetio.readers.icartt import ICARTTReader


@pytest.fixture
def sample_icartt_file(tmp_path):
    icartt_content = """17, 1001
Aero, Jules
NASA
Example Data
AeroMission
1, 1
2023, 07, 01, 2023, 07, 01
1
Time_Start, seconds
2
1, 1
-9999, -9999
Latitude, deg
Longitude, deg
0
1
Comment
1.0, 34.0, -118.0
2.0, 34.1, -118.1
3.0, 34.2, -118.2
"""
    file_path = tmp_path / "test.ict"
    file_path.write_text(icartt_content)
    return str(file_path)


def test_icartt_eager_lazy_consistency(sample_icartt_file):
    reader = ICARTTReader()

    # Eager Mode
    ds_eager = reader.open_dataset(sample_icartt_file, lazy=False)

    # Lazy Mode
    ds_lazy = reader.open_dataset(sample_icartt_file, lazy=True)

    # Verify Lazy is indeed dask-backed
    assert hasattr(ds_lazy.latitude.data, "dask")
    assert hasattr(ds_lazy.time.data, "dask")

    # Compare results (ignoring mesh which is usually just 0)
    ds_eager_no_mesh = ds_eager.drop_vars("mesh", errors="ignore")
    ds_lazy_no_mesh = ds_lazy.drop_vars("mesh", errors="ignore")

    xr.testing.assert_allclose(ds_eager_no_mesh, ds_lazy_no_mesh)

    # Check values
    assert len(ds_eager.node) == 3
    assert ds_eager.latitude.attrs["standard_name"] == "latitude"
    assert ds_eager.attrs["PI"] == "Aero, Jules"
    assert ds_eager.time.values[0] == np.datetime64("2023-07-01T00:00:01")


def test_icartt_missing_values(tmp_path):
    icartt_content = """17, 1001
PI
ORG
SRC
MISS
1, 1
2023, 07, 01, 2023, 07, 01
1
Time, s
1
1
-999
Var1, units
0
2
Comment 1
Comment 2
1.0, 10.0
2.0, -999.0
3.0, 30.0
"""
    file_path = tmp_path / "missing.ict"
    file_path.write_text(icartt_content)

    reader = ICARTTReader()
    ds = reader.open_dataset(str(file_path))

    # Second value should be NaN
    assert np.isnan(ds.Var1.values[1])
    assert ds.Var1.values[0] == 10.0
    assert ds.Var1.values[2] == 30.0
