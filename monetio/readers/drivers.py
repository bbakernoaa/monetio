from typing import Any, Callable, List, Optional, Union

import pandas as pd
import xarray as xr


class PandasDriver:
    """Driver for readers that primarily produce pandas DataFrames."""

    @staticmethod
    def open_dataset(
        files: List[str],
        read_function: Callable,
        post_process_function: Optional[Callable] = None,
        lazy: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, "Any"]:
        """Open multiple files and combine them into a single DataFrame.

        Parameters
        ----------
        files : List[str]
            List of file paths or URLs to read.
        read_function : Callable
            Function to read a single file into a DataFrame.
        post_process_function : Optional[Callable], optional
            Function to post-process the combined DataFrame.
        lazy : bool, optional
            Whether to load the data lazily using Dask.
        **kwargs
            Additional arguments passed to read_function and post_process_function.

        Returns
        -------
        Union[pd.DataFrame, dask.dataframe.DataFrame]
            The combined DataFrame.
        """
        if lazy:
            import dask
            import dask.dataframe as dd

            dfs = [dask.delayed(read_function)(f, **kwargs) for f in files]
            # We use meta inference if possible
            df = dd.from_delayed(dfs)
        else:
            df = pd.concat([read_function(f, **kwargs) for f in files], ignore_index=True)

        if post_process_function is not None:
            df = post_process_function(df, **kwargs)

        return df


class XarrayDriver:
    """Driver for readers that primarily produce xarray Datasets."""

    @staticmethod
    def open_dataset(
        files: Union[str, List[str]],
        preprocess: Optional[Callable] = None,
        lazy: bool = True,
        **kwargs,
    ) -> xr.Dataset:
        """Open multiple files and combine them into a single Dataset.

        Parameters
        ----------
        files : Union[str, List[str]]
            List of file paths or URLs to read, or a wildcard string.
        preprocess : Optional[Callable], optional
            Function to preprocess each dataset before combining.
        lazy : bool, optional
            Whether to load the data lazily using Dask.
        **kwargs
            Additional arguments passed to xr.open_mfdataset or xr.open_dataset.

        Returns
        -------
        xr.Dataset
            The combined Dataset.
        """
        if not lazy:
            # If not lazy, we might still want to open multiple files
            if isinstance(files, str) and ("*" in files or "?" in files):
                import glob

                files = sorted(glob.glob(files))

            if isinstance(files, list):
                ds = xr.open_mfdataset(files, preprocess=preprocess, **kwargs)
                return ds.compute()
            else:
                ds = xr.open_dataset(files, **kwargs)
                if preprocess is not None:
                    ds = preprocess(ds)
                return ds.compute()
        else:
            if isinstance(files, list) or (
                isinstance(files, str) and ("*" in files or "?" in files)
            ):
                return xr.open_mfdataset(files, preprocess=preprocess, chunks="auto", **kwargs)
            else:
                ds = xr.open_dataset(files, chunks="auto", **kwargs)
                if preprocess is not None:
                    ds = preprocess(ds)
                return ds
