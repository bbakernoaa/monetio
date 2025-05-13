from dataclasses import dataclass
from pathlib import Path

import dask.array as da
import pytest

from monetio.models.ufs import open_mfdataset


@dataclass
class SurfOnlyTestData:
    surf_only: bool
    expected_nz: int
    expected_to_be_loaded: tuple[str, ...] = ("dz_m", "surfalt_m", "pres_pa_mid", "alt_msl_m_full")


@pytest.mark.parametrize(
    "test_data",
    [
        SurfOnlyTestData(surf_only=True, expected_nz=1),
        SurfOnlyTestData(surf_only=False, expected_nz=64),
    ],
    ids=lambda x: f"surf_only={x.surf_only}",
)
def test_open_mfdataset_surf_only(data_dir: Path, test_data: SurfOnlyTestData) -> None:
    slug = "aqm.t12z.dyn.f*.nc"
    actual = open_mfdataset(str(data_dir / "ufs" / slug), surf_only=test_data.surf_only)

    for var in actual.data_vars.values():
        shape_dict = {dim: actual.sizes[dim] for dim in var.dims}
        # Assert there is only one level when extracting surface data
        if "z" in shape_dict:
            assert shape_dict["z"] == test_data.expected_nz
        try:
            assert isinstance(var.data, da.Array)
        except AssertionError:
            # Some variables are loaded from disk for pre-processing or calculated at runtime
            assert var.name in test_data.expected_to_be_loaded
