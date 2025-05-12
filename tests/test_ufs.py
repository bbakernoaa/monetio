from pathlib import Path
from typing import Any

import pytest
import xarray

from monetio.models._rrfs_cmaq_mm import open_mfdataset
import monetio
from pytest_mock import MockerFixture


@pytest.mark.parametrize("surf_only", [True, False], ids=lambda x: f"surf_only={x}")
def test_open_mfdataset(data_dir: Path, surf_only: bool, mocker: MockerFixture) -> None:
    print(data_dir)
    slug = "aqm.t12z.dyn.f*.nc"

    spy = mocker.spy(monetio.models._rrfs_cmaq_mm, "_isel_surface_level_")

    actual = open_mfdataset(str(data_dir / "ufs" / slug), surf_only=surf_only)
    print(actual)

    assert spy.call_count == int(surf_only)


# def test_write():
#     fns = ["/opt/project/local-data/aqm.t12z.dyn.f000.nc",
#            "/opt/project/local-data/aqm.t12z.dyn.f001.nc"]
#     outdir = Path("/opt/project/tests/data/ufs")
#
#     for fn in fns:
#         dset = xarray.open_dataset(fn)
#         dset = dset.isel(grid_yt=slice(110, 120), grid_xt=slice(350, 360))
#         print(dset)
#         dset.to_netcdf(outdir / Path(fn).name)