"""HYSPLIT Reader. Deprecated wrapper — use monetio.load('hysplit', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.hysplit import (  # noqa: F401
    HYSPLITReader,
    add_species,
    check_drange,
    check_grid_continuity,
    combine_dataset as combine_dataset_reader,
    fix_grid_continuity,
    get_latlongrid,
    getlatlon,
    open_dataset_hysplit,
    reset_latlon_coords,
)


@deprecated_wrapper(
    "monetio.models.hysplit.open_dataset",
    'monetio.load("hysplit", files=...)',
)
def open_dataset(fname, **kwargs):
    """Method to open HYSPLIT netcdf files.

    Parameters
    ----------
    fname : string or list
        fname is the path to the file or files.
    **kwargs : dict
        Additional arguments passed to HYSPLITReader.open_dataset

    Returns
    -------
    xarray.DataSet
    """
    return HYSPLITReader().open_dataset(files=fname, **kwargs)


@deprecated_wrapper(
    "monetio.models.hysplit.combine_dataset",
    'monetio.load("hysplit", files=...)',
)
def combine_dataset(
    blist,
    drange=None,
    species=None,
    century=None,
    verbose=False,
    sample_time_stamp="start",
    check_grid=True,
):
    """Method to combine multiple HYSPLIT datasets."""
    return combine_dataset_reader(
        blist,
        drange=drange,
        species=species,
        century=century,
        verbose=verbose,
        sample_time_stamp=sample_time_stamp,
        check_grid=check_grid,
    )
