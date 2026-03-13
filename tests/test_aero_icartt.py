import numpy as np
import xarray as xr

from monetio.readers.icartt import ICARTTReader


def create_mock_icartt(filename):
    """Create a mock ICARTT file."""
    # Wait, Latitude/Longitude are DVARs in my mock.
    # Header:
    # 15 lines total
    # 1: 15, 1001
    # 2: PI
    # 3: ORG
    # 4: SRC
    # 5: MIS
    # 6: 1
    # 7: 2023, 05, 15...
    # 8: 1
    # 9: Time_Start, seconds
    # 10: 2 (DVARs)
    # 11: 1.0, 1.0 (Scales)
    # 12: -999.9, -999.9 (Miss)
    # 13: Latitude
    # 14: Longitude
    # 15: Data header

    header = [
        "15, 1001",
        "Aero PI",
        "Aero Org",
        "Aero Source",
        "Aero Mission",
        "1",
        "2023, 05, 15, 2023, 05, 15",
        "1",
        "Time_Start, seconds",
        "2",
        "1.0, 1.0",
        "-999, -999",
        "Latitude, deg",
        "Longitude, deg",
        "Time_Start, Latitude, Longitude",
    ]
    data = ["0, 40.0, -100.0", "1, 41.0, -101.0", "2, -999, -102.0"]
    # Scales: Latitude * 10, Longitude * 0.1
    # Missing: -999.9
    # In data:
    # Row 0: 0, 40.0, -100.0
    # Row 1: 1, 41.0, -101.0
    # Row 2: 2, Miss, -102.0

    with open(filename, "w") as f:
        f.write("\n".join(header + data))


def test_icartt_reader_lazy(tmp_path):
    fn = str(tmp_path / "test.ict")
    create_mock_icartt(fn)

    reader = ICARTTReader()

    # 1. Eager
    ds_eager = reader.open_dataset(fn, lazy=False)
    # If expanded 2D, it should have time and node dims.
    # For a single file with one platform, node=1.
    if "time" in ds_eager.dims:
        lat0 = ds_eager.latitude.isel(time=0).values
        lon0 = ds_eager.longitude.isel(time=0).values
        lat2 = ds_eager.latitude.isel(time=2).values
    else:
        lat0 = ds_eager.latitude.isel(node=0).values
        lon0 = ds_eager.longitude.isel(node=0).values
        lat2 = ds_eager.latitude.isel(node=2).values

    assert lat0 == 40.0
    assert lon0 == -100.0
    assert np.isnan(lat2)

    # 2. Lazy
    ds_lazy = reader.open_dataset(fn, lazy=True)
    # PointReader currently returns Dask-backed variables if lazy=True
    # But wait, PointReader.to_xarray uses to_dask_array(lengths=True)
    assert ds_lazy.latitude.chunks is not None

    # 3. Compare
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())


def test_icartt_metadata(tmp_path):
    fn = str(tmp_path / "test.ict")
    create_mock_icartt(fn)

    reader = ICARTTReader()
    ds = reader.open_dataset(fn)
    assert ds.attrs["PI"] == "Aero PI"
    assert ds.attrs["mission"] == "Aero Mission"
