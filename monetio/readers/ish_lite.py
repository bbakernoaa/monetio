"""ISH Lite Reader"""

import os
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    import dask.dataframe as dd

from ..util import force_object_strings, normalize_pandas_freq
from .base import PointReader, register_reader
from .drivers import FileUtility
from .ish import ISH
from .sat_utils import update_history


def read_ish_lite_file(fname: str, **kwargs) -> pd.DataFrame:
    """
    Read a single ISH (Integrated Surface Hourly) Lite file.

    Parameters
    ----------
    fname : str
        File path, URL, or fsspec-compatible path.
    **kwargs : dict
        Additional arguments passed to fsspec.open.

    Returns
    -------
    pd.DataFrame
        The loaded data in long format.
    """
    from numpy import nan

    columns = [
        "year",
        "month",
        "day",
        "hour",
        "temp",
        "dew_pt_temp",
        "press",
        "wdir",
        "ws",
        "sky_condition",
        "precip_1hr",
        "precip_6hr",
    ]

    # Use FileUtility
    fs = FileUtility.get_fs(fname)
    compression = "gzip" if fname.endswith(".gz") else None
    storage_options = kwargs.get("storage_options", {})

    with fs.open(fname, "rb", compression=compression, **storage_options) as f:
        df = pd.read_csv(
            f,
            sep=r"\s+",
            header=None,
            names=columns,
        )
    # Create time column manually
    df["time"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df.drop(["year", "month", "day", "hour"], axis=1, inplace=True)

    filename = os.path.basename(fname).split("-")
    siteid = filename[0] + filename[1]
    for col in ["temp", "dew_pt_temp", "press", "ws", "precip_1hr", "precip_6hr"]:
        df[col] /= 10.0
    df["siteid"] = siteid
    df = df.replace(-9999, nan)
    return df


@register_reader("ish_lite")
class ISHLiteReader(PointReader):
    def build_urls(
        self,
        dates: Optional[pd.DatetimeIndex] = None,
        box: Optional[List[float]] = None,
        country: Optional[str] = None,
        state: Optional[str] = None,
        site: Optional[str] = None,
        source: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        """
        Build ISH Lite URLs.
        """
        ish = ISH()
        if source is not None:
            ish.source = source

        dates = pd.to_datetime(dates)
        if ish.history is None:
            ish.read_ish_history(dates=dates)
        dfloc = ish.history.copy()

        if box is not None:
            dfloc = ish.subset_sites(latmin=box[0], lonmin=box[1], latmax=box[2], lonmax=box[3])
        elif country is not None:
            dfloc = dfloc.loc[dfloc.ctry == country, :]
        elif state is not None:
            dfloc = dfloc.loc[dfloc.state == state, :]
        elif site is not None:
            dfloc = dfloc.loc[dfloc.station_id == site, :]

        urls = ish.build_urls(dates=dates, sites=dfloc, lite=True)
        if urls.empty:
            return []
        return urls.name.tolist()

    def open_dataset(
        self,
        files: Optional[Union[str, List[str]]] = None,
        dates: Optional[Union[pd.DatetimeIndex, List[datetime], datetime, str]] = None,
        box: Optional[List[float]] = None,
        country: Optional[str] = None,
        state: Optional[str] = None,
        site: Optional[str] = None,
        resample: bool = False,
        window: str = "h",
        n_procs: int = 1,
        verbose: bool = False,
        source: Optional[str] = None,
        as_xarray: bool = True,
        lazy: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, xr.Dataset, "dd.DataFrame"]:
        """
        Retrieve and load ISH (Integrated Surface Hourly) Lite data.

        Parameters
        ----------
        files : Union[str, List[str]], optional
            File path, list of paths, or glob pattern.
        dates : Union[pd.DatetimeIndex, List[datetime], datetime, str], optional
            Dates to retrieve if files are not provided.
        box : List[float], optional
            Bounding box [latmin, lonmin, latmax, lonmax].
        country : str, optional
            Country code to filter sites.
        state : str, optional
            State code to filter sites.
        site : str, optional
            Specific station ID to filter.
        resample : bool, optional
            Whether to resample data to a regular window, by default False.
        window : str, optional
            Resampling window (e.g., 'h'), by default 'h'.
        n_procs : int, optional
            Number of processors for dask compute (if not lazy), by default 1.
        verbose : bool, optional
            Whether to print verbose output, by default False.
        source : str, optional
            Data source: 'ncdc' or 'aws', by default 'aws'.
        as_xarray : bool, optional
            Whether to return an xarray.Dataset, by default True.
        lazy : bool, optional
            Whether to return a dask-backed object (dask.dataframe or xarray with dask),
            by default False.
        **kwargs : dict
            Additional arguments passed to the driver or to_xarray.

        Returns
        -------
        Union[pd.DataFrame, xr.Dataset, dd.DataFrame]
            The loaded ISH Lite data.

        Examples
        --------
        >>> from monetio.readers.ish_lite import ISHLiteReader
        >>> reader = ISHLiteReader()
        >>> ds = reader.open_dataset(dates='2021-08-01', site='72406093721')
        """
        ish = ISH()
        if source is not None:
            ish.source = source

        # Use base class to open via super()
        df = super().open_dataset(
            files,
            dates,
            box=box,
            country=country,
            state=state,
            site=site,
            resample=resample,
            window=window,
            n_procs=n_procs,
            verbose=verbose,
            source=source,
            read_method=read_ish_lite_file,
            as_xarray=False,
            lazy=lazy,
            **kwargs,
        )

        # Filtering
        if dates is not None:
            dates = pd.to_datetime(dates)
            # Use exclusive upper bound to match unit test expectations
            df = df.loc[(df.time >= dates.min()) & (df.time < dates.max())]

        df = df.replace(-999.9, np.nan)

        # Merge with metadata
        if ish.history is None:
            ish.read_ish_history()
        dfloc = ish.history.copy()
        dfloc = force_object_strings(dfloc)

        if lazy:
            import dask.dataframe as dd

            df = df.assign(siteid=df.siteid.astype(object))
            dfloc_dask = dd.from_pandas(dfloc, npartitions=1).assign(
                station_id=lambda x: x.station_id.astype(object)
            )
            df = df.merge(dfloc_dask, how="left", left_on="siteid", right_on="station_id")
        else:
            df["siteid"] = df["siteid"].astype(object)
            df = df.merge(dfloc, how="left", left_on="siteid", right_on="station_id")

        df = df.rename(columns={"ctry": "country"}).drop(columns=["station_id"], errors="ignore")

        df = self.harmonize(df)

        if not lazy and hasattr(df, "compute"):
            df = df.compute(num_workers=n_procs)

        if as_xarray:
            from ..util import ds_to_2d

            # We first convert to 1D UGRID
            ds = self.to_xarray(df, expand2d=False, **kwargs)

            # Metadata variables to preserve
            meta_coords = [
                "country",
                "state",
                "station name",
                "elev(m)",
                "latitude",
                "longitude",
                "siteid",
                "usaf",
                "wban",
            ]

            if resample:
                # Backend-agnostic resampling in xarray
                # To preserve per-site data, we expand to 2D (time, node) before resampling
                pivot = kwargs.get("wide_fmt", kwargs.get("pivot", True))
                ds = ds_to_2d(ds, pivot=pivot, fixed_location=self.fixed_location)

                # Identify metadata variables to preserve (those that don't vary with time)
                metadata = xr.Dataset()
                for c in meta_coords:
                    if c in ds.coords or c in ds.data_vars:
                        val = ds[c]
                        if "time" in val.dims:
                            # If it varies with time, we take the first value per node
                            val = val.isel(time=0, drop=True)
                        metadata[c] = val

                try:
                    ds = (
                        ds.sortby("time")
                        .resample(time=normalize_pandas_freq(window))
                        .mean(numeric_only=True)
                    )
                except Exception:
                    ds = ds.sortby("time").resample(time=normalize_pandas_freq(window)).mean()

                # Restore metadata
                for c in metadata.data_vars:
                    ds[c] = metadata[c]
                for c in metadata.coords:
                    ds.coords[c] = metadata.coords[c]

                # Ensure siteid is correctly linked to node if it's the dimension
                if "siteid" not in ds.coords and "siteid" not in ds.data_vars and "node" in ds.dims:
                    ds.coords["siteid"] = (("node",), ds.node.data)

                # Update history for resampling
                ds = update_history(ds, f"Resampled ISH Lite data to {window} window.")

            else:
                # Now expand to 2D if requested (default is True in PointReader)
                expand2d = kwargs.get("expand2d", True)
                if expand2d:
                    pivot = kwargs.get("wide_fmt", kwargs.get("pivot", True))
                    ds = ds_to_2d(ds, pivot=pivot, fixed_location=self.fixed_location)
                    if (
                        "siteid" not in ds.coords
                        and "siteid" not in ds.data_vars
                        and "node" in ds.dims
                    ):
                        ds.coords["siteid"] = (("node",), ds.node.data)

            # Ensure metadata are coordinates
            ds = ds.set_coords([c for c in meta_coords if c in ds.variables])

            # Update history
            ds = update_history(ds, "Read ISH Lite data.")
            return ds

        if resample:
            if not lazy:
                if not df.empty:
                    df = (
                        df.set_index("time")
                        .groupby("siteid")
                        .resample(normalize_pandas_freq(window))
                        .mean(numeric_only=True)
                        .reset_index()
                    )
                    # Re-join metadata for pandas eager path
                    df = df.merge(
                        dfloc.rename(columns={"ctry": "country"}),
                        how="left",
                        left_on="siteid",
                        right_on="station_id",
                    ).drop(columns=["station_id"], errors="ignore")
            else:
                import warnings

                warnings.warn(
                    "ISHLiteReader: Resampling is currently not supported for lazy DataFrames. "
                    "Convert to xarray (as_xarray=True) for lazy resampling."
                )

        return df


def add_data(
    dates: Union[pd.DatetimeIndex, List[datetime], datetime, str],
    box: Optional[List[float]] = None,
    country: Optional[str] = None,
    state: Optional[str] = None,
    site: Optional[str] = None,
    resample: bool = False,
    window: str = "h",
    n_procs: int = 1,
    verbose: bool = False,
    source: Optional[str] = None,
    as_xarray: bool = True,
    lazy: bool = False,
    **kwargs,
) -> Union[pd.DataFrame, xr.Dataset, "dd.DataFrame"]:
    """
    Retrieve and load ISH Lite data (backward-compatible wrapper).

    Parameters
    ----------
    dates : Union[pd.DatetimeIndex, List[datetime], datetime, str]
        Dates to retrieve.
    box : List[float], optional
        Bounding box [latmin, lonmin, latmax, lonmax].
    country : str, optional
        Country code.
    state : str, optional
        State code.
    site : str, optional
        Station ID.
    resample : bool, optional
        Whether to resample, by default False.
    window : str, optional
        Resampling window, by default 'h'.
    n_procs : int, optional
        Number of processors, by default 1.
    verbose : bool, optional
        Verbose output, by default False.
    source : str, optional
        Data source: 'ncdc' or 'aws', by default 'aws'.
    as_xarray : bool, optional
        Return xarray.Dataset, by default True.
    lazy : bool, optional
        Return dask-backed object, by default False.
    **kwargs : dict
        Additional arguments.

    Returns
    -------
    Union[pd.DataFrame, xr.Dataset, dd.DataFrame]
        The loaded ISH Lite data.
    """
    return ISHLiteReader().open_dataset(
        dates=dates,
        box=box,
        country=country,
        state=state,
        site=site,
        resample=resample,
        window=window,
        n_procs=n_procs,
        verbose=verbose,
        source=source,
        as_xarray=as_xarray,
        lazy=lazy,
        **kwargs,
    )
