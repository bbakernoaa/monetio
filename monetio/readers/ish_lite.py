"""ISH Lite Reader"""

import os
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    import dask.dataframe as dd

from ..util import force_object_strings
from .base import PointReader, register_reader
from .drivers import FileUtility
from .ish import ISH
from .sat_utils import update_history


def read_ish_lite_file(fname: str, **kwargs) -> pd.DataFrame:
    """
    Read a single ISH Lite file.

    Parameters
    ----------
    fname : str
        File path or URL.

    Returns
    -------
    pd.DataFrame
        The loaded data.
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
        as_xarray: bool = True,
        lazy: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, xr.Dataset, "dd.DataFrame"]:
        """
        Retrieve and load ISH (Integrated Surface Hourly) Lite data .

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
        as_xarray : bool, optional
            Whether to return an xarray.Dataset, by default True.
        lazy : bool, optional
            Whether to return a dask-backed object, by default False.
        **kwargs : dict
            Additional arguments.

        Returns
        -------
        Union[pd.DataFrame, xr.Dataset, dd.DataFrame]
            The loaded ISH Lite data.
        """
        ish = ISH()

        if files is None and dates is not None:
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
                raise ValueError("No data URLs found")
            files = urls.name.tolist()

        if not files:
            raise ValueError("Must provide either 'files' or 'dates'.")

        # Use base class to open
        df = super().open_dataset(
            files,
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
            ds = self.to_xarray(df, **kwargs)

            # Preserve metadata in coordinates during resampling
            meta_coords = [
                "country",
                "state",
                "station name",
                "elev(m)",
                "latitude",
                "longitude",
                "siteid",
            ]
            ds = ds.set_coords([c for c in meta_coords if c in ds.data_vars])

            if resample:
                # Backend-agnostic resampling in xarray
                try:
                    ds = ds.resample(time=window).mean(numeric_only=True)
                except TypeError:
                    ds = ds.resample(time=window).mean()

            # Update history
            ds = update_history(ds, "Read ISH Lite data.")
            return ds

        if resample:
            if not lazy:
                if not df.empty:
                    df = (
                        df.set_index("time")
                        .groupby("siteid")
                        .resample(window)
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
    as_xarray: bool = True,
    lazy: bool = False,
    **kwargs,
) -> Union[pd.DataFrame, xr.Dataset, "dd.DataFrame"]:
    """
    Backward-compatible wrapper for ISHLiteReader.open_dataset.
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
        as_xarray=as_xarray,
        lazy=lazy,
        **kwargs,
    )
