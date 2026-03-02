import logging
import xarray as xr

logger = logging.getLogger(__name__)


class PandasDriver:
    """Driver for loading data via pandas/dask.dataframe."""

    def __init__(self):
        pass

    def open_dataset(self, files, dates=None, *, lazy=True, **kwargs):
        """Open a dataset from a list of files or URLs.

        Parameters
        ----------
        files : list of str
            The files or URLs to open.
        dates : array-like, optional
            The dates corresponding to the files.
        lazy : bool, optional
            Whether to load the data lazily via dask.
        **kwargs : dict
            Additional arguments passed to the reader function.

        Returns
        -------
        pandas.DataFrame or dask.dataframe.DataFrame
            The combined data.
        """
        import dask
        import dask.dataframe as dd
        import pandas as pd

        # 1. Expand paths if needed
        # (Assuming FileUtility.expand_paths is available or similar logic)

        # 2. Define the reader function (passed as a kwarg usually)
        reader = kwargs.pop("reader", None)
        if reader is None:
            raise ValueError("A reader function must be provided.")

        # 3. Load files
        if lazy:
            dfs = [dask.delayed(reader)(f, **kwargs) for f in files]
            # Try to get metadata from the first file to avoid full compute
            try:
                meta = reader(files[0], **kwargs)
            except Exception as e:
                logger.warning(f"Could not extract metadata from first file: {e}")
                meta = None
            df = dd.from_delayed(dfs, meta=meta)
        else:
            dfs = [reader(f, **kwargs) for f in files]
            df = pd.concat(dfs, ignore_index=True)

        return df


class XarrayDriver:
    """Driver for loading data via xarray."""

    def __init__(self):
        pass

    def open_dataset(self, files, dates=None, *, lazy=True, **kwargs):
        """Open a dataset from a list of files or URLs.

        Parameters
        ----------
        files : str or list of str
            The files or URLs to open. Supports wildcards.
        dates : array-like, optional
            The dates corresponding to the files.
        lazy : bool, optional
            Whether to load the data lazily via dask.
        **kwargs : dict
            Additional arguments passed to xr.open_mfdataset or xr.open_dataset.

        Returns
        -------
        xarray.Dataset
        """
        if isinstance(files, str) and ("*" in files or "?" in files):
            # Wildcard path
            ds = xr.open_mfdataset(files, chunks={} if lazy else None, **kwargs)
        elif isinstance(files, (list, tuple)):
            # List of files
            ds = xr.open_mfdataset(files, chunks={} if lazy else None, **kwargs)
        else:
            # Single file
            ds = xr.open_dataset(files, chunks={} if lazy else None, **kwargs)

        return ds
