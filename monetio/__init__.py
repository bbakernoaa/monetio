import importlib

from . import grids
from .models import (
    camx,
    chimere,
    cmaq,
    hysplit,
    hytraj,
    ncep_grib,
    pardump,
    raqms,
)
from .obs import (
    aeronet,
    airnow,
    aqs,
    cems,
    crn,
    improve,
    ish,
    ish_lite,
    nadp,
    openaq,
    openaq_v2,
    openaq_v3,
    pams,
)
from .profile import geoms, gml_ozonesonde, icartt, tolnet
from .readers.base import READER_REGISTRY
from .sat import goes

__version__ = "0.2.7"

__all__ = [
    "__version__",
    "load",
    #
    # utility functions
    "rename_latlon",
    "rename_to_monet_latlon",
    "dataset_to_monet",
    "coards_to_netcdf",
    #
    # utility modules
    "grids",
    #
    # point obs
    "airnow",
    "aeronet",
    "aqs",
    "cems",  # TODO: module with add_data
    "crn",
    "improve",  # TODO: module with add_data
    "ish",
    "ish_lite",
    "nadp",
    "openaq",
    "openaq_v2",
    "openaq_v3",
    "pams",
    #
    # profile obs
    "geoms",
    "gml_ozonesonde",
    "icartt",
    "tolnet",
    #
    # satellite obs
    "goes",
    #
    # models
    "camx",
    "cmaq",
    "hysplit",
    "hytraj",
    "icap_mme",
    "ncep_grib",
    "pardump",
    "raqms",
    "chimere",
]

# Map reader names to their module paths for lazy loading
_READER_MODULES = {
    # Models
    "cmaq": ".readers.cmaq",
    "camx": ".readers.camx",
    "hysplit": ".readers.hysplit",
    "hytraj": ".readers.hytraj",
    "icap_mme": ".readers.icap_mme",
    "ncep_grib": ".readers.ncep_grib",
    "pardump": ".readers.pardump",
    "raqms": ".readers.raqms",
    "ufs": ".readers.ufs",
    "wrfchem": ".readers.wrfchem",
    "grib2": ".readers.grib2",
    # Obs
    "airnow": ".readers.airnow",
    "aeronet": ".readers.aeronet",
    "aqs": ".readers.aqs",
    "cems": ".readers.cems",
    "crn": ".readers.crn",
    "improve": ".readers.improve",
    "ish": ".readers.ish",
    "ish_lite": ".readers.ish_lite",
    "nadp": ".readers.nadp",
    "openaq": ".readers.openaq",
    "pams": ".readers.pams",
    "ndbc": ".readers.ndbc",
    "surfrad": ".readers.surfrad",
    "solrad": ".readers.solrad",
    # Profile
    "icartt": ".readers.icartt",
    "tolnet": ".readers.tolnet",
    "geoms": ".readers.geoms",
    "gml_ozonesonde": ".readers.gml_ozonesonde",
    "igra2": ".readers.igra2",
    # Sat
    "goes": ".readers.goes",
    "nesdis_edr_viirs": ".readers.nesdis_edr_viirs",
    "nesdis_eps_viirs": ".readers.nesdis_eps_viirs",
    "modis_ornl": ".readers.modis_ornl",
    "nasa_modis": ".readers.nasa_modis",
    "nesdis_frp": ".readers.nesdis_frp",
    "omps": ".readers.omps",
    "omps_nadir": ".readers.omps_nadir",
    "mopitt": ".readers.mopitt",
    "tempo": ".readers.tempo",
    "tropomi": ".readers.tropomi",
    "merra2": ".readers.merra2",
    "nesdis_viirs_jrr": ".readers.nesdis_viirs_jrr",
    "viirs_jrr": ".readers.nesdis_viirs_jrr",
}


def load(source: str, files=None, **kwargs):
    """
    Universal load function.

    Usage:
        ds = monetio.load("cmaq", files="/path/to/data*.nc")
        df = monetio.load("airnow", files=["2023-01-01", "2023-01-02"])

    Available sources:
        Models: cmaq, camx, hysplit, hytraj, icap_mme, ncep_grib, pardump, raqms, ufs, wrfchem, grib2
        Obs: airnow, aeronet, aqs, cems, crn, improve, ish, ish_lite, nadp, openaq, pams
        Profile: icartt, tolnet, geoms, gml_ozonesonde, igra2
        Sat: goes, nesdis_edr_viirs, nesdis_eps_viirs, modis_ornl, nasa_modis, nesdis_frp, omps, omps_nadir, viirs_jrr
    """
    if source not in READER_REGISTRY:
        if source in _READER_MODULES:
            # Lazy import
            importlib.import_module(_READER_MODULES[source], package="monetio")
        else:
            raise ValueError(
                f"Unknown source '{source}'. Available: {list(_READER_MODULES.keys())}"
            )

    if source not in READER_REGISTRY:
        # Should be registered by now if module was valid
        raise RuntimeError(f"Source '{source}' found in lazy index but failed to register itself.")

    # Instantiate the reader class and open data
    reader_cls = READER_REGISTRY[source]
    reader = reader_cls()

    return reader.open_dataset(files=files, **kwargs)


def rename_latlon(ds):
    """Rename latitude/longitude to ``'lat'``/``'lon'``.

    Parameters
    ----------
    ds : xarray.Dataset

    Returns
    -------
    xarray.Dataset
        Dataset with possibly renamed latitude/longitude.
    """
    if "latitude" in ds.coords:
        return ds.rename({"latitude": "lat", "longitude": "lon"})
    elif "Latitude" in ds.coords:
        return ds.rename({"Latitude": "lat", "Longitude": "lon"})
    elif "Lat" in ds.coords:
        return ds.rename({"Lat": "lat", "Lon": "lon"})
    else:
        return ds


def rename_to_monet_latlon(ds):
    """Rename latitude/longitude to ``'latitude'``/``'longitude'``.

    Parameters
    ----------
    ds : xarray.Dataset

    Returns
    -------
    xarray.Dataset
        Dataset with possibly renamed latitude/longitude.

    See Also
    --------
    rename_latlon : renames to ``'lat'``/``'lon'`` instead
    """
    if "lat" in ds.coords:
        return ds.rename({"lat": "latitude", "lon": "longitude"})
    elif "Latitude" in ds.coords:
        return ds.rename({"Latitude": "latitude", "Longitude": "longitude"})
    elif "Lat" in ds.coords:
        return ds.rename({"Lat": "latitude", "Lon": "longitude"})
    elif "grid_lat" in ds.coords:
        return ds.rename({"grid_lat": "latitude", "grid_lon": "longitude"})
    else:
        return ds


def dataset_to_monet(ds, *, lat_name="lat", lon_name="lon", latlon2d=None):
    """Apply :func:`coards_to_netcdf` if `latlon2d` is False.

    Parameters
    ----------
    ds : xarray.Dataset
    lat_name, lon_name : str
        Current latitude and longitude names in `ds`.
    latlon2d : bool, optional
        If not provided, the value will be detected by examining ``.ndim``
        of the latitude variable.

    Returns
    -------
    xarray.Dataset
    """
    if latlon2d is None:
        ndim_lat = ds[lat_name].ndim
        assert ndim_lat <= 2
        latlon2d = ndim_lat == 2
    # TODO: apply rename_to_monet_latlon ?
    if latlon2d is False:
        ds = coards_to_netcdf(ds, lat_name=lat_name, lon_name=lon_name)
    return ds


def coards_to_netcdf(ds, *, lat_name="lat", lon_name="lon"):
    """Assign 2-D latitude/longitude grid from 1-D latitude/longitude variables,
    setting ``'x'`` and ``'y'`` as 1-D zero-based index arrays.

    Also normalizes the latitude/longitude names to ``'latitude'``/``'longitude'``,
    with dimensions ``('y', 'x')``.

    .. note::
       The name is a reference to the COARDS conventions.

    Parameters
    ----------
    ds : xarray.Dataset
    lat_name, lon_name : str
        Current latitude and longitude names in `ds`.

    Returns
    -------
    xarray.Dataset
    """
    from numpy import arange, meshgrid

    lon = ds[lon_name]
    lat = ds[lat_name]
    assert lon.ndim == lat.ndim == 1
    lons, lats = meshgrid(lon, lat)
    x = arange(len(lon))
    y = arange(len(lat))
    ds = ds.rename({lon_name: "x", lat_name: "y"})
    ds.coords["longitude"] = (("y", "x"), lons)
    ds.coords["latitude"] = (("y", "x"), lats)
    ds["x"] = x
    ds["y"] = y
    ds = ds.set_coords(["latitude", "longitude"])
    return ds
