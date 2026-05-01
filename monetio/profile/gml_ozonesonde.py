"""GML Ozonesonde Reader. Deprecated wrapper — use monetio.load('gml_ozonesonde', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.gml_ozonesonde import GMLOzonesondeReader  # noqa: F401


@deprecated_wrapper(
    "monetio.profile.gml_ozonesonde.add_data",
    'monetio.load("gml_ozonesonde", dates=...)',
)
def add_data(dates, *, location=None, n_procs=1, errors="raise", **kwargs):
    """Retrieve and load GML ozonesonde data as a DataFrame.

    Parameters
    ----------
    dates : sequence of datetime-like
        The period between the min and max (both inclusive)
        will be used to select the files to load.
    location : str or sequence of str, optional
        For example 'Boulder, Colorado'.
    n_procs : int
        For Dask.
    errors : {'raise', 'warn', 'skip'}
        What to do when there is an error reading a file.
    **kwargs : dict
        Additional arguments forwarded to ``GMLOzonesondeReader.open_dataset``.

    Returns
    -------
    pandas.DataFrame
    """
    return GMLOzonesondeReader().open_dataset(
        dates=dates, location=location, n_procs=n_procs, errors=errors, **kwargs
    )
