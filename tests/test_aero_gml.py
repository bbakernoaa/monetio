import pandas as pd
import pytest
import xarray as xr

from monetio.readers.gml_ozonesonde import GMLOzonesondeReader, read_100m


def test_gml_ozonesonde_read_100m(tmp_path):
    # Create a dummy .l100 file
    content = """GML Ozonesonde Data File

    Block 2 (ignored)

    Block 3 (ignored)

    Station: Boulder, CO
    Launch Date: 2023-01-01
    Launch Time: 12:00:00
    Latitude: 40.0
    Longitude: -105.0
    Station Height: 1600 m
    Flight Number: BU123
    Background: 0.02
    Flowrate: 30.0
    RH Corr: 0.5
    Sonde Total O3 (SBUV): 300

    Level   Press    Alt   Pottp   Temp   FtempV   Hum  Ozone  Ozone   Ozone  Ptemp  O3 # DN O3 Res  O3 Uncert
     Num     hPa      km     K      C       C       %    mPa    ppmv   atmcm    C   10^11/cc   DU          %
       1  1000.0   0.100  290.0   15.0    15.0    50.0   2.0   0.020  0.0010  20.0    5.0      300.0       5.0
       2   900.0   1.000  295.0   10.0    10.0    45.0   1.5   0.025  0.0020  18.0    4.0      290.0       4.0
    """
    p = tmp_path / "test.l100"
    p.write_text(content)

    df = read_100m(str(p))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df["siteid"].iloc[0] == "Boulder, Colorado"
    assert df["time"].iloc[0] == pd.Timestamp("2023-01-01 12:00:00")


@pytest.mark.parametrize("lazy", [False, True])
def test_gml_ozonesonde_reader(tmp_path, lazy):
    content = """GML Ozonesonde Data File

    Block 2

    Block 3

    Station: Boulder, CO
    Launch Date: 2023-01-01
    Launch Time: 12:00:00
    Latitude: 40.0
    Longitude: -105.0
    Station Height: 1600 m
    Flight Number: BU123

    Level   Press    Alt   Pottp   Temp   FtempV   Hum  Ozone  Ozone   Ozone  Ptemp  O3 # DN O3 Res  O3 Uncert
     Num     hPa      km     K      C       C       %    mPa    ppmv   atmcm    C   10^11/cc   DU          %
       1  1000.0   0.100  290.0   15.0    15.0    50.0   2.0   0.020  0.0010  20.0    5.0      300.0       5.0
       2   900.0   1.000  295.0   10.0    10.0    45.0   1.5   0.025  0.0020  18.0    4.0      290.0       4.0
    """
    p = tmp_path / "test.l100"
    p.write_text(content)

    reader = GMLOzonesondeReader()
    ds = reader.open_dataset(files=str(p), lazy=lazy, as_xarray=True, expand2d=True)

    assert isinstance(ds, xr.Dataset)
    if lazy:
        assert ds.o3.chunks is not None

    # Check dimensions
    assert "lev" in ds.dims or "lev" in ds.coords
    assert "time" in ds.coords
    # siteid should be either siteid or renamed to node
    assert "siteid" in ds.coords or "node" in ds.dims

    # Verify history
    assert "history" in ds.attrs
    assert "Read GML Ozonesonde data" in ds.attrs["history"]
