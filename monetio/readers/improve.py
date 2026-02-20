"""IMPROVE Reader"""

from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, List, Union

import pandas as pd

from ..util import force_object_strings
from .base import PointReader, register_reader
from .drivers import FileUtility
from .epa_utils import read_monitor_file

if TYPE_CHECKING:
    import dask.dataframe as dd
    import xarray as xr


@register_reader("improve")
class IMPROVEReader(PointReader):
    def open_dataset(
        self,
        files: Union[str, List[str]],
        add_meta: bool = False,
        delimiter: str = "\t",
        as_xarray: bool = True,
        lazy: bool = False,
        pivot: bool = True,
        **kwargs,
    ) -> Union[pd.DataFrame, "xr.Dataset", "dd.DataFrame"]:
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
        **kwargs : dict
            Additional arguments passed to the reader and driver.

        Returns
        -------
        Union[pd.DataFrame, xr.Dataset, dd.DataFrame]
            The loaded IMPROVE data.
        """
        # Use PandasDriver via base class
        # Pass a partial of read_improve_file as the read_method
        read_func = partial(read_improve_file, delimiter=delimiter)

        # Pop kwargs that are not for the driver/reader_func
        # but for to_xarray or this method.
        # super().open_dataset will pass **kwargs to the driver.
        # The driver will pass them to read_func.
        # So we should only keep driver-related kwargs if any.
        # Actually, base.PointReader.open_dataset also uses some kwargs.

        driver_kwargs = kwargs.copy()
        for k in ["expand2d", "pivot"]:
            driver_kwargs.pop(k, None)

        df = super().open_dataset(
            files,
            read_method=read_func,
            as_xarray=False,
            lazy=lazy,
            **driver_kwargs,
        )

        # Determine backend
        try:
            import dask.dataframe as dd

            is_dask = isinstance(df, dd.DataFrame)
        except ImportError:
            is_dask = False

        if is_dask:
            if df.npartitions == 0:
                return df
        elif len(df) == 0:
            return df

        if add_meta:
            df = self.add_metadata(df)

        df = self.harmonize(df)

        if as_xarray:
            ds = self.to_xarray(df, pivot=pivot, **kwargs)
            # Update history
            history = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read IMPROVE data."
            if "history" in ds.attrs:
                ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
            else:
                ds.attrs["history"] = history
            return ds

        return df

    def add_metadata(
        self, df: Union[pd.DataFrame, "dd.DataFrame"]
    ) -> Union[pd.DataFrame, "dd.DataFrame"]:
        """
        Add site metadata from the IMPROVE monitor file.

        Parameters
        ----------
        df : Union[pd.DataFrame, "dd.DataFrame"]
            Input dataframe.

        Returns
        -------
        Union[pd.DataFrame, "dd.DataFrame"]
            Dataframe with metadata merged.
        """
        try:
            import dask.dataframe as dd

            is_dask = isinstance(df, dd.DataFrame)
        except ImportError:
            is_dask = False

        monitor_df = read_monitor_file(network="IMPROVE")

        # Ensure siteid is object for reliable merging
        monitor_df = monitor_df.copy().drop_duplicates(subset=["siteid"])
        monitor_df["siteid"] = monitor_df["siteid"].astype(object)

        if is_dask:
            df["epaid"] = df["epaid"].astype(object)
            monitor_dask = dd.from_pandas(monitor_df, npartitions=1)
            df = df.merge(monitor_dask, left_on="epaid", right_on="siteid", how="left")
        else:
            df["epaid"] = df["epaid"].astype(object)
            df = df.merge(monitor_df, left_on="epaid", right_on="siteid", how="left")

        # Handle column name conflicts from merge
        if "siteid_x" in df.columns:
            df = df.drop(columns=["siteid_y", "state_name_y"], errors="ignore")
            df = df.rename(columns={"siteid_x": "siteid", "state_name_x": "state_name"})

        return df


def read_improve_file(fname: str, delimiter: str = "\t", **kwargs) -> pd.DataFrame:
    """
    Read a single IMPROVE data file.

    Parameters
    ----------
    fname : str
        File path or URL.
    delimiter : str, optional
        Delimiter used in the file, by default "\\t".
    **kwargs : dict
        Additional arguments passed to pd.read_csv.

    Returns
    -------
    pd.DataFrame
        The loaded data.
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
        # Fallback or let pd.read_csv fail if file is truly broken
        pass

    # Read the CSV - only pass valid read_csv kwargs if they exist in kwargs
    # For now, we pop the ones we know might be there from open_dataset but are not for read_csv
    for k in ["pivot", "add_meta", "as_xarray", "lazy", "expand2d"]:
        kwargs.pop(k, None)

    df = pd.read_csv(
        fname,
        delimiter=delimiter,
        parse_dates=["Date"],
        dtype={"EPACode": str},
        skiprows=skiprows,
        storage_options=storage_options,
        **{k: v for k, v in kwargs.items() if k != "storage_options"},
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
