"""NESDIS FRP Reader. Deprecated wrapper — use monetio.load('nesdis_frp', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.nesdis_frp import NESDISFRPReader  # noqa: F401


@deprecated_wrapper(
    "monetio.sat.nesdis_frp.download_data",
    'monetio.load("nesdis_frp", dates=...)',
)
def download_data(date, ftype="meanFRP", **kwargs):
    """Download NESDIS FRP data for a given date.

    Parameters
    ----------
    date : str or datetime-like
        Date to retrieve.
    ftype : str
        File type (e.g. 'meanFRP').
    **kwargs : dict
        Additional arguments forwarded to ``NESDISFRPReader.open_dataset``.
    """
    return NESDISFRPReader().open_dataset(
        dates=date, ftype=ftype, **kwargs
    )
