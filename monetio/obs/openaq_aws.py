"""
OpenAQ AWS Reader. Redirection to monetio.readers.openaq_aws
"""

from ..readers.openaq_aws import OpenAQAWSReader, read


def add_data(
    dates,
    *,
    siteid=None,
    country=None,
    provider=None,
    find_paths=True,
    n_procs=1,
):
    """Retrieve and load OpenAQ archive data from AWS."""
    return OpenAQAWSReader().open_dataset(
        dates=dates,
        siteid=siteid,
        country=country,
        provider=provider,
        find_paths=find_paths,
        n_procs=n_procs,
    )
