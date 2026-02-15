"""
OpenAQ AWS Reader. Redirection to monetio.readers.openaq_aws
"""

from ..readers.openaq_aws import (  # noqa: F401
    OpenAQAWSReader,
    _build_urls,
    get_locations,
    get_paths,
    get_provider_countries,
    get_providers,
    read,
)


def add_data(
    dates,
    *,
    siteid=None,
    country=None,
    provider=None,
    find_paths=True,
    n_procs=1,
    as_xarray=True,
):
    """Retrieve and load OpenAQ archive data from AWS."""
    return OpenAQAWSReader().open_dataset(
        dates=dates,
        siteid=siteid,
        country=country,
        provider=provider,
        find_paths=find_paths,
        n_procs=n_procs,
        as_xarray=as_xarray,
    )
