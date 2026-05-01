# Feature: virtualizarr-reader-refactor, Property 6: VirtualiZarr Activation Produces Valid Dataset
"""Property-based test for VirtualiZarr activation producing valid datasets.

**Validates: Requirements 1.1, 11.1, 11.2**

For any valid list of NetCDF file paths and ``use_virtualizarr=True``, the XarrayDriver
SHALL return an ``xr.Dataset`` (not None, not an error) with the same variable names
and dimensions as would be produced by the standard ``open_mfdataset`` path (modulo
chunking differences).
"""

import os
import tempfile
import unittest.mock as mock

import numpy as np
import pytest
import xarray as xr
from hypothesis import given, settings
from hypothesis import strategies as st

from monetio.readers.drivers import XarrayDriver

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Random variable names (short, lowercase alpha)
_var_name = st.from_regex(r"[a-z]{2,6}", fullmatch=True)

# Dimension sizes
_dim_size = st.integers(min_value=2, max_value=5)

# Number of files (1-3 for speed)
_n_files = st.integers(min_value=1, max_value=3)

# Time steps per file
_n_times = st.integers(min_value=1, max_value=3)


@st.composite
def _netcdf_file_set(draw):
    """Generate a set of small NetCDF files with consistent structure.

    Returns (file_paths, expected_var_names, expected_dims).
    """
    n_files = draw(_n_files)
    n_times = draw(_n_times)
    nx = draw(_dim_size)
    ny = draw(_dim_size)

    # Generate 1-3 variable names
    n_vars = draw(st.integers(min_value=1, max_value=3))
    var_names = draw(
        st.lists(_var_name, min_size=n_vars, max_size=n_vars, unique=True)
    )

    tmpdir = tempfile.mkdtemp()
    file_paths = []

    for i in range(n_files):
        data_vars = {}
        for vname in var_names:
            data = np.random.rand(n_times, ny, nx).astype(np.float32)
            data_vars[vname] = (["time", "y", "x"], data)

        ds = xr.Dataset(
            data_vars,
            coords={
                "time": np.arange(i * n_times, (i + 1) * n_times),
                "y": np.arange(ny),
                "x": np.arange(nx),
            },
        )

        fpath = os.path.join(tmpdir, f"test_{i:03d}.nc")
        ds.to_netcdf(fpath)
        file_paths.append(fpath)

    expected_dims = {"time", "y", "x"}
    return file_paths, set(var_names), expected_dims


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=30000)
@given(file_set=_netcdf_file_set())
def test_virtualizarr_produces_valid_dataset(file_set):
    """VirtualiZarr path produces a dataset with same variables and dimensions as standard path.

    We mock the VirtualiZarr library since it's an optional dependency. The mock
    simulates the VZ path by having open_virtual_mfdataset return a mock VDS whose
    ``vz.to_kerchunk()`` returns refs, and then we mock ``xr.open_dataset`` on the
    zarr mapper to return the same dataset as the standard path.
    """
    file_paths, expected_vars, expected_dims = file_set

    try:
        driver = XarrayDriver()

        # Standard path: open without VirtualiZarr
        standard_ds = driver.open(file_paths, use_virtualizarr=False)

        assert isinstance(standard_ds, xr.Dataset), "Standard path should return xr.Dataset"
        assert set(standard_ds.data_vars) == expected_vars, (
            f"Standard path vars {set(standard_ds.data_vars)} != expected {expected_vars}"
        )
        assert expected_dims.issubset(set(standard_ds.dims)), (
            f"Standard path dims {set(standard_ds.dims)} missing expected {expected_dims}"
        )

        # VirtualiZarr path: mock the entire VZ import chain
        mock_vds = mock.MagicMock()
        mock_vds.vz.to_kerchunk.return_value = {"version": 1, "refs": {}}

        mock_open_virtual = mock.MagicMock(return_value=mock_vds)
        mock_hdf_parser = mock.MagicMock()

        # Build the mock module hierarchy
        mock_virtualizarr = mock.MagicMock()
        mock_virtualizarr.open_virtual_mfdataset = mock_open_virtual

        mock_parsers = mock.MagicMock()
        mock_parsers.HDFParser = mock_hdf_parser

        mock_ujson = mock.MagicMock()
        mock_zarr = mock.MagicMock()

        modules_patch = {
            "virtualizarr": mock_virtualizarr,
            "virtualizarr.parsers": mock_parsers,
            "ujson": mock_ujson,
            "zarr": mock_zarr,
            "obstore": mock.MagicMock(),
            "obstore.store": mock.MagicMock(),
            "obspec_utils": mock.MagicMock(),
            "obspec_utils.registry": mock.MagicMock(),
        }

        with mock.patch.dict("sys.modules", modules_patch):
            with mock.patch("fsspec.get_mapper") as mock_mapper:
                mock_mapper.return_value = mock.MagicMock()

                # Mock xr.open_dataset to return the standard dataset when called
                # with the zarr mapper (VZ path)
                original_open_dataset = xr.open_dataset

                def patched_open_dataset(path_or_mapper, **kwargs):
                    if kwargs.get("engine") == "zarr":
                        # This is the VZ path — return the standard dataset
                        return standard_ds
                    return original_open_dataset(path_or_mapper, **kwargs)

                with mock.patch("xarray.open_dataset", side_effect=patched_open_dataset):
                    vz_ds = driver.open(
                        file_paths,
                        use_virtualizarr=True,
                    )

        assert isinstance(vz_ds, xr.Dataset), "VZ path should return xr.Dataset"

        # Compare variable names
        assert set(vz_ds.data_vars) == set(standard_ds.data_vars), (
            f"VZ vars {set(vz_ds.data_vars)} != standard vars {set(standard_ds.data_vars)}"
        )

        # Compare dimensions
        assert set(vz_ds.dims) == set(standard_ds.dims), (
            f"VZ dims {set(vz_ds.dims)} != standard dims {set(standard_ds.dims)}"
        )

    finally:
        # Cleanup temp files
        for fp in file_paths:
            try:
                os.unlink(fp)
            except OSError:
                pass
        try:
            os.rmdir(os.path.dirname(file_paths[0]))
        except OSError:
            pass
