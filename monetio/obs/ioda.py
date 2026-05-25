def add_data(files, **kwargs):
    """
    Load IODA data.

    Parameters
    ----------
    files : str or list of str
        File path(s), URL(s), or glob pattern.
    **kwargs
        Additional arguments passed to the IODAReader.

    Returns
    -------
    xr.Dataset
    """
    from ..readers.ioda import IODAReader

    return IODAReader().open_dataset(files=files, **kwargs)
