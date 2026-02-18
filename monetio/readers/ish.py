"""ISH Reader"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    import dask.dataframe as dd

from .base import PointReader, register_reader
from .drivers import FileUtility

logger = logging.getLogger(__name__)


class ISH:
    """Helper class for ISH data retrieval."""

    VAR_INFO = [
        ("varlength", "i2", 4),
        ("station_id", "S11", 11),
        ("date", "i4", 8),
        ("htime", "i2", 4),
        ("source_flag", "S1", 1),
        ("latitude", "float", 6),
        ("longitude", "float", 7),
        ("code", "S5", 5),
        ("elev", "i2", 5),
        ("call_letters", "S5", 5),
        ("qc_process", "S4", 4),
        ("wdir", "i2", 3),
        ("wdir_quality", "S1", 1),
        ("wdir_type", "S1", 1),
        ("ws", "i2", 4),
        ("ws_quality", "S1", 1),
        ("ceiling", "i4", 5),
        ("ceiling_quality", "S1", 1),
        ("ceiling_code", "S1", 1),
        ("ceiling_cavok", "S1", 1),
        ("vsb", "i4", 6),
        ("vsb_quality", "S1", 1),
        ("vsb_variability", "S1", 1),
        ("vsb_variability_quality", "S1", 1),
        ("t", "i2", 5),
        ("t_quality", "S1", 1),
        ("dpt", "i2", 5),
        ("dpt_quality", "S1", 1),
        ("p", "i4", 5),
        ("p_quality", "S1", 1),
    ]

    DTYPES = [(name, dtype) for name, dtype, _ in VAR_INFO]
    WIDTHS = [width for _, _, width in VAR_INFO]

    def __init__(self):
        self.history_file = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
        self.history = None
        self.dates = None
        self.verbose = False
        self.source = "ncdc"

    def read_ish_history(self, dates: Optional[pd.DatetimeIndex] = None):
        """
        Read the ISH history file.

        Parameters
        ----------
        dates : pd.DatetimeIndex, optional
            Dates to filter the history, by default None.
        """
        if dates is None:
            dates = self.dates
        fname = self.history_file

        if self.source == "aws":
            fname = "s3://noaa-isd-pds/isd-history.csv"

        fs = FileUtility.get_fs(fname)
        try:
            with fs.open(fname, "r") as f:
                self.history = pd.read_csv(
                    f, parse_dates=["BEGIN", "END"], dtype={"USAF": str, "WBAN": str}
                )
        except Exception:
            alt = fname.replace("www1.ncdc.noaa.gov", "www.ncei.noaa.gov")
            if alt != fname:
                fs_alt = FileUtility.get_fs(alt)
                with fs_alt.open(alt, "r") as f:
                    self.history = pd.read_csv(
                        f, parse_dates=["BEGIN", "END"], dtype={"USAF": str, "WBAN": str}
                    )
                self.history_file = alt
            else:
                raise

        self.history.columns = [i.lower() for i in self.history.columns]
        if dates is not None:
            index1 = (self.history.end >= dates.min()) & (self.history.begin <= dates.max())
            self.history = self.history.loc[index1, :]
        self.history = self.history.dropna(subset=["lat", "lon"])
        self.history.loc[:, "usaf"] = self.history.usaf.astype("str").str.zfill(6)
        self.history.loc[:, "wban"] = self.history.wban.astype("str").str.zfill(5)
        self.history["station_id"] = self.history.usaf + self.history.wban
        self.history.rename(columns={"lat": "latitude", "lon": "longitude"}, inplace=True)

    def subset_sites(
        self,
        latmin: float = 32.65,
        lonmin: float = -113.3,
        latmax: float = 34.5,
        lonmax: float = -110.4,
    ) -> pd.DataFrame:
        """
        Subset sites by bounding box.
        """
        latindex = (self.history.latitude >= latmin) & (self.history.latitude <= latmax)
        lonindex = (self.history.longitude >= lonmin) & (self.history.longitude <= lonmax)
        dfloc = self.history.loc[latindex & lonindex, :]
        return dfloc

    def read_data_frame(self, url_or_file, **kwargs):
        """
        Legacy method for backward compatibility.
        """
        df = read_ish_file(url_or_file, **kwargs)
        return df

    def build_urls(
        self,
        dates: Optional[pd.DatetimeIndex] = None,
        sites: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Construct ISH URLs.
        """
        if dates is None:
            dates = self.dates
        if sites is None:
            sites = self.history

        unique_years = pd.to_datetime(dates.year.unique(), format="%Y")
        furls = []

        if self.source == "aws":
            url = "s3://noaa-isd-pds/data"
        else:
            url = "https://www.ncei.noaa.gov/pub/data/noaa"

        for syear in unique_years.strftime("%Y"):
            # USAF is 6 digits, WBAN is 5 digits
            year_fnames = (
                sites.usaf.astype(str).str.zfill(6)
                + "-"
                + sites.wban.astype(str).str.zfill(5)
                + "-"
                + syear
                + ".gz"
            )
            for fname in year_fnames:
                furls.append(f"{url}/{syear}/{fname}")

        return pd.Series(furls, name="name").to_frame()

    def get_url_file_objs(self, fname):
        import gzip
        import shutil

        import requests

        objs = []
        for iii in fname:
            try:
                r2 = requests.get(iii, stream=True)
                if r2.status_code != 404:
                    temp = iii.split("/")[-1]
                    out_name = "isd." + temp.replace(".gz", "")
                    objs.append(out_name)
                    with open(out_name, "wb") as fid:
                        gzip_file = gzip.GzipFile(fileobj=r2.raw)
                        shutil.copyfileobj(gzip_file, fid)
            except Exception:
                pass
        return objs


def _clean_col(
    series: pd.Series, missing_vals: Union[float, List[float]], multiplier: float = 1.0
) -> pd.Series:
    """
    Clean a numeric column by replacing missing values with NaN and applying a multiplier.
    """
    if not isinstance(missing_vals, (list, tuple)):
        missing_vals = [missing_vals]
    series = series.astype(float)
    for mv in missing_vals:
        series = series.where(series != mv, np.nan)
    return series / multiplier


def read_ish_file(fname: str, **kwargs) -> pd.DataFrame:
    """
    Read a single ISH (Integrated Surface Hourly) file.

    Parameters
    ----------
    fname : str
        File path or URL.
    **kwargs : dict
        Additional arguments.

    Returns
    -------
    pd.DataFrame
        The loaded data.
    """
    import fsspec

    # Regression fix: check valid retries
    request_retries = kwargs.get("request_retries", 4)
    if request_retries < 0:
        raise ValueError(f"`request_retries` must be >= 0, got {request_retries!r}")

    request_timeout = kwargs.get("request_timeout", 10)

    compression = "gzip" if str(fname).endswith(".gz") else None

    # Implement a simple retry loop for legacy compatibility in tests
    frame_as_array = None
    tries = 0
    while tries <= request_retries:
        try:
            if str(fname).startswith("http") or str(fname).startswith("ftp"):
                import io

                import requests

                r = requests.get(fname, timeout=request_timeout, stream=True)
                r.raise_for_status()
                content = r.content
                if compression == "gzip":
                    import gzip

                    with gzip.open(io.BytesIO(content), "rb") as f:
                        frame_as_array = np.genfromtxt(f, delimiter=ISH.WIDTHS, dtype=ISH.DTYPES)
                else:
                    frame_as_array = np.genfromtxt(
                        io.BytesIO(content), delimiter=ISH.WIDTHS, dtype=ISH.DTYPES
                    )
            else:
                with fsspec.open(fname, "rb", compression=compression) as f:
                    frame_as_array = np.genfromtxt(f, delimiter=ISH.WIDTHS, dtype=ISH.DTYPES)
            break
        except Exception as e:
            tries += 1
            if tries > request_retries:
                if "timeout" in str(e).lower() or "connect" in str(e).lower() or tries > 1:
                    raise RuntimeError(f"Failed to connect to server for URL {fname}.") from e
                logger.warning(f"Could not read {fname}: {e}")
                return pd.DataFrame()

    if frame_as_array is None:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(np.atleast_1d(frame_as_array))

    if df.empty:
        return df

    # Vectorized cleaning
    # Time construction
    dt_str = df["date"].astype(str).str.zfill(8) + df["htime"].astype(str).str.zfill(4)
    df["time"] = pd.to_datetime(dt_str, format="%Y%m%d%H%M", errors="coerce")
    df = df.dropna(subset=["time"])

    # Decode bytes
    for col, dtype in ISH.DTYPES:
        if "S" in str(dtype) and col in df.columns:
            df[col] = df[col].str.decode("utf-8").str.strip()

    # Numeric cleaning
    df["wdir"] = _clean_col(df["wdir"], 999)
    df["ws"] = _clean_col(df["ws"], 9999, multiplier=10.0)
    df["ceiling"] = _clean_col(df["ceiling"], 99999)
    df["vsb"] = _clean_col(df["vsb"], [99999, 999999])
    df["t"] = _clean_col(df["t"], 9999, multiplier=10.0)
    df["dpt"] = _clean_col(df["dpt"], 9999, multiplier=10.0)
    df["p"] = _clean_col(df["p"], 99999, multiplier=10.0)

    df = df.drop(columns=["date", "htime", "latitude", "longitude"], errors="ignore")

    return df


@register_reader("ish")
class ISHReader(PointReader):
    def open_dataset(
        self,
        files: Optional[Union[str, List[str]]] = None,
        dates: Optional[Union[pd.DatetimeIndex, List[datetime], datetime, str]] = None,
        box: Optional[List[float]] = None,
        country: Optional[str] = None,
        state: Optional[str] = None,
        site: Optional[str] = None,
        resample: bool = True,
        window: str = "h",
        download: bool = False,
        n_procs: int = 1,
        verbose: bool = False,
        source: str = "ncdc",
        as_xarray: bool = True,
        lazy: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, xr.Dataset, "dd.DataFrame"]:
        """
        Retrieve and load ISH (Integrated Surface Hourly) data following the Aero Protocol.

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
            Whether to resample data to a regular window, by default True.
        window : str, optional
            Resampling window (e.g., 'h'), by default 'h'.
        download : bool, optional
            Whether to download files (if source is ncdc), by default False.
        n_procs : int, optional
            Number of processors for dask compute (if not lazy), by default 1.
        verbose : bool, optional
            Whether to print verbose output, by default False.
        source : str, optional
            Data source: 'ncdc' or 'aws', by default 'ncdc'.
        as_xarray : bool, optional
            Whether to return an xarray.Dataset, by default True.
        lazy : bool, optional
            Whether to return a dask-backed object, by default False.
        **kwargs : dict
            Additional arguments.

        Returns
        -------
        Union[pd.DataFrame, xr.Dataset, dd.DataFrame]
            The loaded ISH data.
        """
        # Regression fix: check multiple subsets
        if sum([box is not None, country is not None, state is not None, site is not None]) > 1:
            raise ValueError("Only one of `box`, `country`, `state`, or `site` can be used")

        ish = ISH()
        ish.source = source

        if files is None and dates is not None:
            dates = pd.to_datetime(dates)
            if ish.history is None:
                ish.read_ish_history(dates=dates)
            dfloc_urls = ish.history.copy()

            if box is not None:
                dfloc_urls = ish.subset_sites(
                    latmin=box[0], lonmin=box[1], latmax=box[2], lonmax=box[3]
                )
            elif country is not None:
                dfloc_urls = dfloc_urls.loc[dfloc_urls.ctry == country, :]
            elif state is not None:
                dfloc_urls = dfloc_urls.loc[dfloc_urls.state == state, :]
            elif site is not None:
                dfloc_urls = dfloc_urls.loc[dfloc_urls.station_id == site, :]

            urls = ish.build_urls(dates=dates, sites=dfloc_urls)
            if urls.empty:
                raise ValueError("No data URLs found")
            files = urls.name.tolist()

        if files is None:
            raise ValueError("Must provide either 'files' or 'dates'.")

        if download:
            files = ish.get_url_file_objs(files)

        # Use driver directly to avoid extra harmonize calls that might clash
        df = self.driver.open(files, read_method=read_ish_file, lazy=lazy, **kwargs)

        # Filtering by date if requested
        if dates is not None:
            dates = pd.to_datetime(dates)
            df = df.loc[(df.time >= dates.min()) & (df.time <= dates.max())]

        # Construct siteid
        if "station_id" in df.columns:
            df["siteid"] = df["station_id"]

        # Merge with metadata
        if ish.history is None:
            ish.read_ish_history()
        dfloc = ish.history.copy()

        if lazy:
            import dask.dataframe as dd

            df = df.assign(siteid=df.siteid.astype(object))
            dfloc_dask = dd.from_pandas(
                dfloc.rename(columns={"station_id": "siteid"}), npartitions=1
            ).assign(siteid=lambda x: x.siteid.astype(object))
            df = df.merge(dfloc_dask, on="siteid", how="left")
        else:
            df["siteid"] = df["siteid"].astype(object)
            df = df.merge(dfloc.rename(columns={"station_id": "siteid"}), on="siteid", how="left")

        df = df.rename(columns={"ctry": "country"})

        df = self.harmonize(df)

        if not lazy and hasattr(df, "compute"):
            df = df.compute(num_workers=n_procs)

        if as_xarray:
            # We first convert to 1D
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
                from ..util import ds_to_2d

                pivot = kwargs.get("wide_fmt", kwargs.get("pivot", True))
                ds = ds_to_2d(ds, pivot=pivot)

                # Identify metadata variables to preserve
                metadata = xr.Dataset()
                for c in meta_coords:
                    if c in ds.coords or c in ds.data_vars:
                        val = ds[c]
                        if "time" in val.dims:
                            val = val.isel(time=0, drop=True)
                        metadata[c] = val

                try:
                    ds = ds.sortby("time").resample(time=window).mean(numeric_only=True)
                except Exception:
                    ds = ds.sortby("time").resample(time=window).mean()

                # Restore metadata
                for c in metadata.data_vars:
                    ds[c] = metadata[c]
                for c in metadata.coords:
                    ds.coords[c] = metadata.coords[c]
                if "siteid" not in ds.coords and "siteid" not in ds.data_vars and "node" in ds.dims:
                    ds.coords["siteid"] = (("node",), ds.node.values)

                ds = ds.set_coords([c for c in meta_coords if c in ds.data_vars])

            else:
                # Now expand to 2D if requested (default is True in PointReader)
                expand2d = kwargs.get("expand2d", True)
                if expand2d:
                    from ..util import ds_to_2d

                    pivot = kwargs.get("wide_fmt", kwargs.get("pivot", True))
                    ds = ds_to_2d(ds, pivot=pivot)
                    if (
                        "siteid" not in ds.coords
                        and "siteid" not in ds.data_vars
                        and "node" in ds.dims
                    ):
                        ds.coords["siteid"] = (("node",), ds.node.values)

            # Update history
            history = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read ISH data."
            if "history" in ds.attrs:
                ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
            else:
                ds.attrs["history"] = history
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
                    # Re-join metadata
                    meta_to_restore = dfloc.rename(
                        columns={"ctry": "country", "station_id": "siteid"}
                    )
                    # Avoid duplicate columns during merge
                    cols_to_drop = [
                        c for c in meta_to_restore.columns if c in df.columns and c != "siteid"
                    ]
                    df = df.drop(columns=cols_to_drop)
                    df = df.merge(meta_to_restore, on="siteid", how="left")
            else:
                import warnings

                warnings.warn(
                    "ISHReader: Resampling is currently not supported for lazy DataFrames. "
                    "Convert to xarray (as_xarray=True) for lazy resampling."
                )

        return df
