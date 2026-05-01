"""MODIS ORNL Reader. Deprecated wrapper — use monetio.load('modis_ornl', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.modis_ornl import MODISORNLReader  # noqa: F401


@deprecated_wrapper(
    "monetio.sat.modis_ornl.open_dataset",
    'monetio.load("modis_ornl", files=...)',
)
def open_dataset(
    date,
    product="MOD12A2H",
    band="Lai_500m",
    quality_control=None,
    latitude=0,
    longitude=0,
    kmAboveBelow=100,
    kmLeftRight=100,
    **kwargs,
):
    """Open MODIS data from ORNL DAAC web service.

    Parameters
    ----------
    date : str or datetime-like
        Date to retrieve.
    product : str
        MODIS product name.
    band : str
        MODIS band name.
    quality_control : str, optional
        Quality control band name.
    latitude : float
        Latitude of the point of interest.
    longitude : float
        Longitude of the point of interest.
    kmAboveBelow : int
        Km above and below the point.
    kmLeftRight : int
        Km left and right of the point.
    **kwargs : dict
        Additional arguments forwarded to ``MODISORNLReader.open_dataset``.

    Returns
    -------
    xarray.DataArray
    """
    return MODISORNLReader().open_dataset(
        dates=date,
        product=product,
        band=band,
        quality_control=quality_control,
        latitude=latitude,
        longitude=longitude,
        kmAboveBelow=kmAboveBelow,
        kmLeftRight=kmLeftRight,
        **kwargs,
    )


@deprecated_wrapper(
    "monetio.sat.modis_ornl.open_mfdataset",
    'monetio.load("modis_ornl", files=...)',
)
def open_mfdataset(
    dates,
    product="MOD12A2H",
    band="Lai_500m",
    quality_control=None,
    latitude=0,
    longitude=0,
    kmAboveBelow=100,
    kmLeftRight=100,
    **kwargs,
):
    """Open multiple MODIS data from ORNL DAAC web service.

    Parameters
    ----------
    dates : sequence of datetime-like
        Dates to retrieve.
    product : str
        MODIS product name.
    band : str
        MODIS band name.
    quality_control : str, optional
        Quality control band name.
    latitude : float
        Latitude of the point of interest.
    longitude : float
        Longitude of the point of interest.
    kmAboveBelow : int
        Km above and below the point.
    kmLeftRight : int
        Km left and right of the point.
    **kwargs : dict
        Additional arguments forwarded to ``MODISORNLReader.open_dataset``.

    Returns
    -------
    xarray.DataArray
    """
    return MODISORNLReader().open_dataset(
        dates=dates,
        product=product,
        band=band,
        quality_control=quality_control,
        latitude=latitude,
        longitude=longitude,
        kmAboveBelow=kmAboveBelow,
        kmLeftRight=kmLeftRight,
        **kwargs,
    )
