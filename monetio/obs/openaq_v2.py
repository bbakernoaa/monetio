"""OpenAQ v2 Reader. Deprecated wrapper — use monetio.load('openaq_v2', ...) instead.

Visit https://docs.openaq.org/docs/getting-started to get an API key
and set environment variable ``OPENAQ_API_KEY`` to use it.

For example, in Bash:

.. code-block:: bash

   export OPENAQ_API_KEY="your_api_key_here"

https://openaq.org/

https://api.openaq.org/docs#/v2
"""

from ..readers._deprecation import deprecated_wrapper
from ..readers.openaq_v2 import (  # noqa: F401
    OpenAQV2Reader,
    _api_key_warning,
    _consume,
    get_locations as _get_locations,
    get_parameters as _get_parameters,
)


@deprecated_wrapper(
    "monetio.obs.openaq_v2.add_data",
    'monetio.load("openaq_v2", dates=...)',
)
def add_data(
    dates,
    *,
    parameters=None,
    country=None,
    search_radius=None,
    sites=None,
    entity=None,
    sensor_type=None,
    query_time_split="1h",
    wide_fmt=False,
    **kwargs,
):
    """Get OpenAQ API v2 data, including low-cost sensors.

    Parameters
    ----------
    dates : datetime-like or array-like of datetime-like
        One desired date/time or
        an array, of which the min and max will be used
        as inclusive time bounds of the desired data.
    parameters : str or list of str, optional
        For example, ``'o3'`` or ``['pm25', 'o3']`` (default).
    country : str or list of str, optional
        For example, ``'US'`` or ``['US', 'CA']`` (two-letter country codes).
    search_radius : dict, optional
        Mapping of coords tuple (lat, lon) [deg] to search radius [m].
    sites : list of str, optional
        Site ID(s) to include.
    entity : str or list of str, optional
        Options: ``'government'``, ``'research'``, ``'community'``.
    sensor_type : str or list of str, optional
        Options: ``'low-cost sensor'``, ``'reference grade'``.
    query_time_split
        Frequency to use when splitting the web API queries in time.
    wide_fmt : bool
        Convert dataframe to wide format (one column per parameter).
    **kwargs
        Additional keyword arguments passed to the reader.
    """
    return OpenAQV2Reader().open_dataset(
        dates=dates,
        parameters=parameters,
        country=country,
        sites=sites,
        wide_fmt=wide_fmt,
        search_radius=search_radius,
        entity=entity,
        sensor_type=sensor_type,
        query_time_split=query_time_split,
        **kwargs,
    )


@deprecated_wrapper(
    "monetio.obs.openaq_v2.get_locations",
    'from monetio.readers.openaq_v2 import get_locations',
)
def get_locations(**kwargs):
    """Get available site info (including site IDs) from OpenAQ v2 API.

    kwargs are passed to :func:`_consume`.

    https://api.openaq.org/docs#/v2/locations_get_v2_locations_get
    """
    return _get_locations(**kwargs)


@deprecated_wrapper(
    "monetio.obs.openaq_v2.get_parameters",
    'from monetio.readers.openaq_v2 import get_parameters',
)
def get_parameters(**kwargs):
    """Get supported parameter info from OpenAQ v2 API.

    kwargs are passed to :func:`_consume`.
    """
    return _get_parameters(**kwargs)


@deprecated_wrapper(
    "monetio.obs.openaq_v2.get_latlonbox_sites",
    'from monetio.readers.openaq_v2 import get_locations',
)
def get_latlonbox_sites(latlonbox, **kwargs):
    """From all available sites, return those within a lat/lon box.

    kwargs are passed to :func:`_consume`.

    Parameters
    ----------
    latlonbox : array-like of float
        ``[lat1, lon1, lat2, lon2]`` (lower-left corner, upper-right corner)
    """
    lat1, lon1, lat2, lon2 = latlonbox
    sites = _get_locations(**kwargs)

    in_box = (
        (sites.latitude >= lat1)
        & (sites.latitude <= lat2)
        & (sites.longitude >= lon1)
        & (sites.longitude <= lon2)
    )

    return sites[in_box].reset_index(drop=True)
