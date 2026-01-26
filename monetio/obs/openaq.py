"""
OpenAQ Reader. Redirection to monetio.readers.openaq
"""

from ..readers.openaq import OPENAQ, OpenAQReader, read_json


def add_data(dates, *, n_procs=1, wide_fmt=True):
    """Retrieve and load OpenAQ data as a DataFrame."""
    return OpenAQReader().open_dataset(dates=dates, n_procs=n_procs, wide_fmt=wide_fmt)
