import re

import numpy as np
import pytest
import xarray as xr

from monetio.readers.cmaq import CMAQReader, cmaq_preprocess


@pytest.fixture
def sample_cmaq_dataset():
    """Create a synthetic CMAQ IOAPI Dataset for testing."""
    times = np.array([2024001, 2024001], dtype=np.int32)
    tstep = np.array([0, 10000], dtype=np.int32)  # 00:00:00 and 01:00:00
    tflag = np.zeros((2, 3, 2), dtype=np.int32)
    tflag[:, :, 0] = times[:, None]
    tflag[:, :, 1] = tstep[:, None]

    o3_data = np.full((2, 1, 4, 5), 0.04, dtype=np.float32)  # 0.04 ppmV
    no2_data = np.full((2, 1, 4, 5), 0.01, dtype=np.float32)  # 0.01 ppmV

    ds = xr.Dataset(
        data_vars={
            "TFLAG": (("TSTEP", "VAR", "DATE-TIME"), tflag),
            "O3": (("TSTEP", "LAY", "ROW", "COL"), o3_data, {"units": "ppmV", "var_desc": "O3"}),
            "NO2": (("TSTEP", "LAY", "ROW", "COL"), no2_data, {"units": "ppmV", "var_desc": "NO2"}),
        },
        attrs={
            "GDTYP": 2,  # LCC
            "P_ALP": 30.0,
            "P_BET": 60.0,
            "P_GAM": -97.0,
            "CENT_MER": -97.0,
            "LAT_TE": 40.0,
            "XCENT": -97.0,
            "YCENT": 40.0,
            "XORIG": -100000.0,
            "YORIG": -100000.0,
            "XCELL": 12000.0,
            "YCELL": 12000.0,
            "NCOLS": 5,
            "NROWS": 4,
            "NLAYS": 1,
            "NVARS": 2,
            "VAR-LIST": "O3              NO2             ",
            "FILEDESC": "Synthetic CMAQ Test Data",
            "history": "Initial creation",
        },
    )
    return ds


def _strip_timestamps(ds: xr.Dataset) -> xr.Dataset:
    """Strip timestamp strings from history attributes for deterministic assertion."""
    ds_clean = ds.copy(deep=True)
    if "history" in ds_clean.attrs:
        # Replace YYYY-MM-DD HH:MM:SS with fixed timestamp
        ds_clean.attrs["history"] = re.sub(
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", "TIMESTAMP", ds_clean.attrs["history"]
        )
    return ds_clean


def test_cmaq_preprocess_eager_lazy_parity(sample_cmaq_dataset):
    """Verify that cmaq_preprocess produces identical output for Eager (NumPy) and Lazy (Dask) backends."""
    ds_eager = sample_cmaq_dataset.copy(deep=True)
    res_eager = cmaq_preprocess(ds_eager, convert_to_ppb=True)

    ds_lazy = sample_cmaq_dataset.copy(deep=True).chunk({"TSTEP": 1, "ROW": 2, "COL": 2})
    res_lazy = cmaq_preprocess(ds_lazy, convert_to_ppb=True)

    # Assert identical structure and values after normalizing timestamp string in history
    xr.testing.assert_identical(_strip_timestamps(res_eager), _strip_timestamps(res_lazy))


def test_cmaq_harmonize_eager_lazy_parity(sample_cmaq_dataset):
    """Verify that CMAQReader.harmonize produces identical output for Eager (NumPy) and Lazy (Dask) backends."""
    reader = CMAQReader()

    ds_eager = sample_cmaq_dataset.copy(deep=True)
    res_eager = reader.harmonize(ds_eager)

    ds_lazy = sample_cmaq_dataset.copy(deep=True).chunk({"TSTEP": 1, "ROW": 2, "COL": 2})
    res_lazy = reader.harmonize(ds_lazy)

    # Assert identical structure and values after normalizing timestamp string in history
    xr.testing.assert_identical(_strip_timestamps(res_eager), _strip_timestamps(res_lazy))
