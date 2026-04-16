"""
HYSPLIT Reader. Redirection to monetio.readers.hysplit
"""

from ..readers.hysplit import (  # noqa: F401  # noqa: F401
    HYSPLITReader,
    add_species,
    check_drange,
    check_grid_continuity,
    combine_dataset as combine_dataset_reader,  # noqa: F401
    fix_grid_continuity,
    get_latlongrid,
    getlatlon,
    open_dataset_hysplit,
    reset_latlon_coords,
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
