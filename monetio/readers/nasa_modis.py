"""NASA MODIS Reader"""

import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader


@register_reader("nasa_modis")
class NASAMODISReader(GriddedReader):
    def open_dataset(self, files, **kwargs):
        """
        Reads NASA MODIS swath data.
        """
        # Expand paths via driver logic but we open file by file due to specific logic
        # Actually, GriddedReader driver can open them.
        # But we need _get_swath_from_fname logic on each file.

        # If multiple files, we likely want to concat? Or return list?
        # Standard open_mfdataset might not work if they are swath (not gridded same way)?

        # Original logic was 'open_single_file'.

        # Let's iterate files.
        # Use FileUtility from drivers if needed, or rely on user passing valid list.
        from .drivers import FileUtility

        file_list = FileUtility.expand_paths(files)

        dsets = []
        for f in file_list:
            ds = open_single_file(f)
            dsets.append(ds)

        if not dsets:
            return xr.Dataset()

        if len(dsets) == 1:
            return dsets[0]
        else:
            return xr.concat(dsets, dim="time")


# -----------------------------------------------------------------------------
# Helper functions ported from monetio/sat/nasa_modis.py
# -----------------------------------------------------------------------------


def _get_swath_from_fname(fname):
    vert_grid_num = fname.split(".")[-4].split("v")[-1]
    hori_grid_num = fname.split(".")[-4].split("v")[0].split("h")[-1]
    return hori_grid_num, vert_grid_num


def _get_time_from_fname(fname):
    u = pd.Series([fname.split(".")[-2]])
    date = pd.to_datetime(u, format="%Y%j%H%M%S")[0]
    return date


def open_single_file(fname):
    from monetio.grids import get_modis_latlon_from_swath_hv, get_sinu_area_def

    h, v = _get_swath_from_fname(fname)
    timestamp = _get_time_from_fname(fname)
    dset = xr.open_dataset(fname)
    dset = dset.rename({"XDim:MOD_Grid_BRDF": "x", "YDim:MOD_Grid_BRDF": "y"})
    dset = get_modis_latlon_from_swath_hv(h, v, dset)
    dset.attrs["area"] = get_sinu_area_def(dset)
    dset["time"] = timestamp
    return dset
