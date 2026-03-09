from typing import Any, List, Optional, Sequence, Union

import pandas as pd
import xarray as xr

from .core import PointReader


class PandasDriver:
    """Driver for reading data using Pandas or Dask."""

    def open_dataset(
        self,
        files: List[str],
        reader: PointReader,
        dates: Optional[Sequence[Any]] = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open dataset with Pandas or Dask.

        Parameters
        ----------
        files : list of str
            List of file paths to read.
        reader : PointReader
            The reader object with a read_data method.
        dates : sequence of datetime-like, optional
            Dates of interest.
        **kwargs : dict
            Additional arguments to pass to the reader.

        Returns
        -------
        xr.Dataset
            The loaded dataset.
        """
        lazy = kwargs.pop("lazy", False)
        # removed unused n_procs

        if lazy:
            import dask
            import dask.dataframe as dd

            # Eagerly sample the first file to determine the metadata for Dask
            first_df = reader.read_data(files[0:1], **kwargs)
            meta = first_df.iloc[:0]  # Get an empty DataFrame with the same schema

            # Use dask.delayed to parallelize the reading
            dfs = [dask.delayed(reader.read_data)([f], **kwargs) for f in files]
            df = dd.from_delayed(dfs, meta=meta)
        else:
            df = reader.read_data(files, **kwargs)

        # Handle dates filtering if provided
        if dates is not None:
            # Vectorized filtering on Dask or Pandas
            df = df.loc[
                (df.time >= pd.to_datetime(dates.min())) & (df.time <= pd.to_datetime(dates.max()))
            ]

        return reader.to_xarray(df, **kwargs)


class XarrayDriver:
    """Driver for reading data using Xarray."""

    def open_dataset(
        self, files: Union[str, List[str]], lazy: bool = False, **kwargs: Any
    ) -> xr.Dataset:
        """Open dataset with Xarray.

        Parameters
        ----------
        files : list of str
            List of file paths or wildcards to read.
        lazy : bool, optional
            Whether to open the dataset lazily with Dask.
        **kwargs : dict
            Additional arguments to pass to xr.open_mfdataset or xr.open_dataset.

        Returns
        -------
        xr.Dataset
            The loaded dataset.
        """
        if isinstance(files, str) and ("*" in files or "?" in files):
            if lazy:
                return xr.open_mfdataset(files, **kwargs)
            else:
                return xr.open_mfdataset(files, combine="by_coords", **kwargs).load()
        else:
            if lazy:
                return xr.open_mfdataset(files, **kwargs)
            else:
                return xr.open_mfdataset(files, combine="by_coords", **kwargs).load()
