"""GOES Satellite Reader. Deprecated wrapper — use monetio.load('goes', ...) instead.

The ``add_goes_bands`` utility function is **not** deprecated and remains
available here.
"""

import xarray as xr

from ..readers._deprecation import deprecated_wrapper
from ..readers.goes import GOESReader  # noqa: F401


@deprecated_wrapper(
    "monetio.sat.goes.open_dataset",
    'monetio.load("goes", files=...)',
)
def open_dataset(date=None, filename=None, satellite="16", product=None, **kwargs):
    """Open GOES data from Amazon S3 or a local file.

    Parameters
    ----------
    date : str or datetime-like, optional
        Date to retrieve from S3.
    filename : str, optional
        Local file path. If provided, ``date`` is ignored.
    satellite : str, optional
        Satellite identifier (e.g. '16', '17', '18').
    product : str, optional
        GOES product (e.g. 'ABI-L2-AODF').
    **kwargs : dict
        Additional arguments forwarded to ``GOESReader.open_dataset``.

    Returns
    -------
    xarray.Dataset
    """
    reader = GOESReader()
    if filename is not None:
        return reader.open_dataset(files=filename, satellite=satellite, **kwargs)
    else:
        return reader.open_dataset(
            dates=date, satellite=satellite, product=product or "ABI-L2-AODF", **kwargs
        )


# ---------------------------------------------------------------------------
# Utility function — NOT deprecated
# ---------------------------------------------------------------------------


def add_goes_bands(
    dset: xr.Dataset,
    blue_band: str = "blue",
    red_band: str = "red",
    veggie_band: str = "veggie",
) -> xr.Dataset:
    """Makes true color image from GOES-R satellite. Must have blue, red, veggie bands.

    Note: This function modifies the input Dataset in-place by adding the 'tci' variable.

    Parameters
    ----------
    dset : xarray.Dataset
        needs to have at least blue, red, veggie bands as data variables.
    blue_band : str
        Name of the blue band variable in the dataset.
    red_band : str
        Name of the red band variable in the dataset.
    veggie_band : str
        Name of the veggie band variable in the dataset.

    Returns
    -------
    xarray.Dataset
        the original dataset with the true color image array added.
    """
    # make green band
    green = 0.45 * dset[red_band] + 0.1 * dset[veggie_band] + 0.45 * dset[blue_band]

    # Get the dimensions from one of the input bands
    dims = dset[red_band].dims

    # Create the true color image DataArray
    # Stack the bands along a new 'rgb' dimension
    tci = xr.concat([dset[red_band], green, dset[blue_band]], dim="rgb").transpose(
        *(dims + ("rgb",))
    )

    # add to the dataset
    dset["tci"] = tci
    dset["tci"].attrs = {
        "long_name": "GOES-R True Color Image",
        "standard_name": "tci",
    }

    return dset
