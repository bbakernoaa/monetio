"""IMPROVE Reader"""

from functools import partial
from typing import Any, List, Union

import pandas as pd
import xarray as xr

from ..util import force_object_strings
from .base import PointReader, register_reader
from .drivers import FileUtility
from .epa_utils import read_monitor_file
from .sat_utils import update_history

try:
    import dask.dataframe as dd
except ImportError:
    dd = None


@register_reader("improve")
class IMPROVEReader(PointReader):
    """
    Reader for IMPROVE (Interagency Monitoring of Protected Visual Environments) data.
    """

    def open_dataset(
        self,
        files: Union[str, List[str]],
        add_meta: bool = False,
        delimiter: str = "\t",
        as_xarray: bool = True,
        lazy: bool = False,
        pivot: bool = True,
        **kwargs: Any,
    ) -> Union[pd.DataFrame, xr.Dataset, "dd.DataFrame"]:
        """
        Retrieve and load IMPROVE data.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path, list of paths, or glob pattern.
        add_meta : bool, optional
            Whether to add site metadata, by default False.
        delimiter : str, optional
            Delimiter used in the IMPROVE file, by default "\\t".
        as_xarray : bool, optional
            Whether to return an xarray.Dataset, by default True.
        lazy : bool, optional
            Whether to return a dask-backed object, by default False.
        pivot : bool, optional
            Whether to pivot the data to wide format, by default True.
        **kwargs : Any
            Additional arguments passed to the reader and driver.

        Returns
        -------
        Union[pd.DataFrame, xr.Dataset, dd.DataFrame]
            The loaded IMPROVE data.

        Examples
        --------
        >>> reader = IMPROVEReader()
        >>> ds = reader.open_dataset("improve_data.txt")
        """
        # Use PandasDriver via base class
        read_func = partial(read_improve_file, delimiter=delimiter)

        driver_kwargs = kwargs.copy()
        for k in ["expand2d", "pivot", "add_meta", "as_xarray", "lazy"]:
            driver_kwargs.pop(k, None)

        df = super().open_dataset(
            files,
            read_method=read_func,
            as_xarray=False,
            lazy=lazy,
            **driver_kwargs,
        )

        # Check for empty (Backend-agnostic)
        is_empty = False
        if dd is not None and isinstance(df, dd.DataFrame):
            if df.npartitions == 0:
                is_empty = True
        elif len(df) == 0:
            is_empty = True

        if is_empty:
            if as_xarray:
                return xr.Dataset()
            return df

        if add_meta:
            df = self.add_metadata(df)

        df = self.harmonize(df)

        if as_xarray:
            ds = self.to_xarray(df, pivot=pivot, **kwargs)
            # Update history
            ds = update_history(ds, "Read IMPROVE data.")
            return ds

        return df

    def add_metadata(
        self, df: Union[pd.DataFrame, "dd.DataFrame"]
    ) -> Union[pd.DataFrame, "dd.DataFrame"]:
        """
        Add site metadata from the IMPROVE monitor file.

        Parameters
        ----------
        df : Union[pd.DataFrame, dd.DataFrame]
            Input dataframe.

        Returns
        -------
        Union[pd.DataFrame, dd.DataFrame]
            Dataframe with metadata merged.

        Examples
        --------
        >>> df = reader.add_metadata(df)
        """
        monitor_df = read_monitor_file(network="IMPROVE")

        # Ensure siteid is object for reliable merging
        monitor_df = monitor_df.copy().drop_duplicates(subset=["siteid"])
        monitor_df = force_object_strings(monitor_df)

        # Backend-agnostic site ID cast
        df["epaid"] = df["epaid"].astype(object)

        # Identify backend and wrap monitor_df if needed
        if hasattr(df, "npartitions"):
            # Dask detected
            import dask.dataframe as dd_local

            monitor_wrap = dd_local.from_pandas(monitor_df, npartitions=1)
        else:
            monitor_wrap = monitor_df

        # Merge
        df = df.merge(monitor_wrap, left_on="epaid", right_on="siteid", how="left")

        # Handle column name conflicts from merge
        if "siteid_x" in df.columns:
            df = df.drop(columns=["siteid_y", "state_name_y"], errors="ignore")
            df = df.rename(columns={"siteid_x": "siteid", "state_name_x": "state_name"})

        # Update history if possible
        df = update_history(df, "Merged with IMPROVE station metadata.")

        return df


def read_improve_file(fname: str, delimiter: str = "\t", **kwargs: Any) -> pd.DataFrame:
    """
    Read a single IMPROVE data file.

    Parameters
    ----------
    fname : str
        File path or URL.
    delimiter : str, optional
        Delimiter used in the file, by default "\\t".
    **kwargs : Any
        Additional arguments passed to pd.read_csv.

    Returns
    -------
    pd.DataFrame
        The loaded data.

    Examples
    --------
    >>> df = read_improve_file("site_data.txt")
    """
    fs = FileUtility.get_fs(fname)

    # Determine storage options if S3
    storage_options = kwargs.get("storage_options")
    if fname.startswith("s3://") and storage_options is None:
        storage_options = {"anon": True}

    # Find the data section
    skiprows = 0
    try:
        with fs.open(fname, "r") as f:
            for i, line in enumerate(f):
                if line.strip() == "Data":
                    skiprows = i + 1
                    break
    except Exception:
        pass

    # Read the CSV
    read_kwargs = kwargs.copy()
    for k in ["pivot", "add_meta", "as_xarray", "lazy", "expand2d", "storage_options"]:
        read_kwargs.pop(k, None)

    df = pd.read_csv(
        fname,
        delimiter=delimiter,
        parse_dates=["Date"],
        dtype={"EPACode": str},
        skiprows=skiprows,
        storage_options=storage_options,
        **read_kwargs,
    )

    # Standardize columns
    df = df.rename(
        columns={
            "EPACode": "epaid",
            "Val": "obs",
            "State": "state_name",
            "ParamCode": "variable",
            "SiteCode": "siteid",
            "Unit": "units",
            "Date": "time",
        }
    )

    if "Dataset" in df.columns:
        df = df.drop(columns="Dataset")

    df.columns = [i.lower() for i in df.columns]

    if "epaid" in df.columns:
        df["epaid"] = df["epaid"].astype(str).str.zfill(9)

    return force_object_strings(df)
