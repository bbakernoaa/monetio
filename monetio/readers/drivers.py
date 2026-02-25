from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Optional, Union

import pandas as pd

if TYPE_CHECKING:
    import dask.dataframe as dd
    import xarray as xr

class PandasDriver:
    """Driver for loading tabular data via Pandas or Dask.

    Follows the Aero Protocol:
    - Supports Eager (Pandas) and Lazy (Dask) backends.
    - Standardized open_dataset signature.
    """

    def open_dataset(
        self,
        files: str | Iterable[str],
        dates: Optional[Iterable[Any]] = None,
        *,
        lazy: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame | dd.DataFrame:
        """Open one or more files as a DataFrame.

        Parameters
        ----------
        files : str or list of str
            Path(s) to the file(s) to load. Supports wildcards.
        dates : array-like, optional
            The dates corresponding to the files.
        lazy : bool, optional
            If True, use Dask to load the data lazily. Default is True.
        **kwargs
            Additional arguments passed to read_csv.

        Returns
        -------
        pandas.DataFrame or dask.dataframe.DataFrame
        """
        if lazy:
            import dask.dataframe as dd
            # dask.dataframe.read_csv handles wildcards and lists of files
            df = dd.read_csv(files, **kwargs)
        else:
            if isinstance(files, str):
                df = pd.read_csv(files, **kwargs)
            else:
                df = pd.concat([pd.read_csv(f, **kwargs) for f in files], ignore_index=True)

        return df

class XarrayDriver:
    """Driver for loading gridded data via Xarray.

    Follows the Aero Protocol.
    """

    def open_dataset(
        self,
        files: str | Iterable[str],
        dates: Optional[Iterable[Any]] = None,
        *,
        lazy: bool = True,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open one or more files as an xarray Dataset.

        Parameters
        ----------
        files : str or list of str
            Path(s) to the file(s) to load. Supports wildcards.
        dates : array-like, optional
            The dates corresponding to the files.
        lazy : bool, optional
            If True, use Dask (via chunks) to load the data lazily. Default is True.
        **kwargs
            Additional arguments passed to open_dataset or open_mfdataset.

        Returns
        -------
        xarray.Dataset
        """
        import xarray as xr

        if lazy:
            # If chunks is not in kwargs, we provide a default to ensure it's lazy
            if "chunks" not in kwargs:
                kwargs["chunks"] = "auto"
        else:
            # Ensure chunks is None for eager loading
            kwargs["chunks"] = None

        if isinstance(files, str) and ("*" in files or "?" in files):
            ds = xr.open_mfdataset(files, **kwargs)
        elif isinstance(files, list) and len(files) > 1:
            ds = xr.open_mfdataset(files, **kwargs)
        else:
            # Single file
            if isinstance(files, list):
                files = files[0]
            ds = xr.open_dataset(files, **kwargs)

        return ds
