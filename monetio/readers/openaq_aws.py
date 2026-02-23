"""OpenAQ archive data on AWS.

https://openaq.org/
https://registry.opendata.aws/openaq/
https://docs.openaq.org/aws/about
"""

import logging
import warnings
from datetime import datetime
from typing import TYPE_CHECKING, List, Union

import pandas as pd
import xarray as xr

from .sat_utils import update_history

if TYPE_CHECKING:
    import dask.dataframe as dd

from .base import PointReader, register_reader

logger = logging.getLogger(__name__)


@register_reader("openaq_aws")
class OpenAQAWSReader(PointReader):
    """OpenAQ AWS Archive Reader"""

    def open_dataset(
        self,
        files: Union[str, List[str]] = None,
        dates: Union[pd.DatetimeIndex, List[datetime], datetime, str] = None,
        siteid: Union[str, List[str]] = None,
        country: Union[str, List[str]] = None,
        provider: Union[str, List[str]] = None,
        find_paths: bool = True,
        wide_fmt: bool = False,
        n_procs: int = 1,
        as_xarray: bool = True,
        lazy: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, xr.Dataset, "dd.DataFrame"]:
        """
        Add OpenAQ data from AWS Open Data.

        Parameters
        ----------
        files : Union[str, List[str]], optional
            File path, list of paths, or glob pattern.
        dates : Union[pd.DatetimeIndex, List[datetime], datetime, str], optional
            Dates to retrieve if files are not provided.
        siteid : Union[str, List[str]], optional
            Site ID(s) to filter by.
        country : Union[str, List[str]], optional
            Country code(s) to filter by.
        provider : Union[str, List[str]], optional
            Provider name(s) to filter by.
        find_paths : bool, optional
            Whether to find paths via S3 listing (slow), by default True.
        wide_fmt : bool, optional
            Whether to return data in wide format, by default False.
        n_procs : int, optional
            Number of processors for dask compute, by default 1.
        as_xarray : bool, optional
            Whether to return an xarray.Dataset, by default True.
        lazy : bool, optional
            Whether to return a dask-backed object, by default False.
        **kwargs : dict
            Additional arguments.

        Returns
        -------
        Union[pd.DataFrame, xr.Dataset, dd.DataFrame]
            The loaded dataset.
        """

        # For backward compatibility, if the first argument looks like dates, swap them.
        if (
            files is not None
            and dates is None
            and isinstance(files, (pd.DatetimeIndex, datetime, pd.Timestamp, list, str))
        ):
            if isinstance(files, (pd.DatetimeIndex, datetime, pd.Timestamp)):
                dates = files
                files = None
            elif isinstance(files, list) and len(files) > 0 and isinstance(files[0], datetime):
                dates = files
                files = None

        read_func = read_openaq_aws_csv

        if files is None and dates is not None:
            dates = _to_datetime_index(dates).dropna()
            if dates.empty:
                raise ValueError("must provide at least one datetime-like")

            if find_paths:
                paths = get_paths(dates, siteid=siteid, country=country, provider=provider)
                files = [f"s3://{p}" for p in paths]
            else:
                if siteid is None:
                    raise ValueError("must provide `siteid` when `find_paths` is false")
                files = build_urls(dates, siteid)
                read_func = read_openaq_aws_csv_robust

        if not files:
            # Handle empty
            if lazy:
                import dask.dataframe as dd

                df = dd.from_pandas(pd.DataFrame(), npartitions=1)
            else:
                df = pd.DataFrame()
            if as_xarray:
                return xr.Dataset()
            return df

        # Use base class to open
        df = super().open_dataset(
            files,
            read_method=read_func,
            as_xarray=False,
            lazy=lazy,
            **kwargs,
        )

        # Post-processing
        df = self.harmonize(df)

        if wide_fmt:
            from ..util import long_to_wide

            df = long_to_wide(df)

        if not lazy and hasattr(df, "compute"):
            df = df.compute(num_workers=n_procs)

        if as_xarray:
            ds = self.to_xarray(df, expand2d=wide_fmt, **kwargs)

            # Update history
            ds = update_history(ds, "Read OpenAQ AWS data.")
            return ds

        return df


def read_openaq_aws_csv(fp: str, **kwargs) -> pd.DataFrame:
    """
    Read OpenAQ archive data from a file-like object.

    Parameters
    ----------
    fp : str
        File path or URL.
    **kwargs : dict
        Additional arguments passed to pd.read_csv.

    Returns
    -------
    pd.DataFrame
        OpenAQ archive data.
    """
    df = pd.read_csv(
        fp,
        dtype={
            0: str,  # location_id
            1: str,  # sensor_id
            2: str,  # location
            3: str,  # datetime
            4: float,  # lat
            5: float,  # lon
            6: str,  # parameter
            7: str,  # unit
            8: float,  # value
        },
        parse_dates=["datetime"],
        **kwargs,
    )

    # Normalize to web API column names
    if "sensors_id" in df.columns:
        df = df.rename(columns={"sensors_id": "sensor_id"})
    if "unit" in df.columns:
        df = df.rename(columns={"unit": "units"})
    df = df.rename(
        columns={
            "location_id": "siteid",
            "datetime": "time",
            "lat": "latitude",
            "lon": "longitude",
        }
    )

    # Convert to UTC, non-localized
    if not df.empty:
        if df["time"].dt.tz is not None:
            df["time"] = df["time"].dt.tz_convert("UTC").dt.tz_localize(None)

    return df


def read_openaq_aws_csv_robust(fp: str, **kwargs) -> pd.DataFrame:
    """Try to read a file, returning empty DF if not found."""
    try:
        return read_openaq_aws_csv(fp, **kwargs)
    except FileNotFoundError:
        return pd.DataFrame(
            columns=[
                "siteid",
                "sensor_id",
                "location",
                "time",
                "latitude",
                "longitude",
                "parameter",
                "units",
                "value",
            ]
        )


def _maybe_to_list(x, *, not_none=False):
    """Convert non-None scalar to singleton list, or return original."""
    if not_none:
        assert x is not None
    if x is not None and pd.api.types.is_scalar(x):
        return [x]
    else:
        return x


def _to_datetime_index(dates, **kwargs):
    """Convert `dates` to a pandas DatetimeIndex."""
    dates = pd.to_datetime(dates, **kwargs)
    if pd.api.types.is_scalar(dates):
        dates = pd.DatetimeIndex([dates])

    return dates


def get_paths(dates, *, siteid=None, country=None, provider=None):
    """Get site-day paths, searching independently by location ID, country, and provider."""
    import s3fs

    fs = s3fs.S3FileSystem(anon=True)

    dates = _to_datetime_index(dates)
    location_ids = _maybe_to_list(siteid)
    providers = _maybe_to_list(provider)
    countries = _maybe_to_list(country)

    if location_ids is None and providers is None and countries is None:
        warnings.warn(
            "location ID(s) not provided; using all locations, which may be quite slow",
            stacklevel=2,
        )
        location_ids = ["*"]

    unique_dates = dates.floor("D").unique()

    paths = []

    if location_ids is not None:
        tpl = (
            "openaq-data-archive/records/csv.gz/"
            "locationid={loc}/year={date:%Y}/month={date:%m}/"
            "location-{loc}-{date:%Y%m%d}.csv.gz"
        )
        for date in unique_dates:
            for loc in location_ids:
                glb = tpl.format(loc=loc, date=date)
                if "*" in glb:
                    loc_date_paths = fs.glob(glb)
                    paths.extend(loc_date_paths)
                else:
                    if fs.exists(glb):
                        paths.append(glb)

    if providers is not None:
        tpl = (
            "openaq-data-archive/records/csv.gz/"
            "provider={prvdr}/country=*/locationid={loc}/"
            "year={date:%Y}/month={date:%m}/"
            "location-{loc}-{date:%Y%m%d}.csv.gz"
        )
        for date in unique_dates:
            for prvdr in providers:
                glb = tpl.format(prvdr=prvdr.lower(), loc="*", date=date)
                prvdr_date_paths = fs.glob(glb)
                paths.extend(prvdr_date_paths)

    if countries is not None:
        tpl = (
            "openaq-data-archive/records/csv.gz/"
            "provider=*/country={cntry}/locationid={loc}/"
            "year={date:%Y}/month={date:%m}/"
            "location-{loc}-{date:%Y%m%d}.csv.gz"
        )
        for date in unique_dates:
            for cntry in countries:
                glb = tpl.format(cntry=cntry.lower(), loc="*", date=date)
                cntry_date_paths = fs.glob(glb)
                paths.extend(cntry_date_paths)

    return sorted(set(paths))


def get_providers():
    """Get OpenAQ data providers by searching the bucket paths."""
    import s3fs

    fs = s3fs.S3FileSystem(anon=True)
    paths = fs.glob("openaq-data-archive/records/csv.gz/provider=*", maxdepth=1)
    providers = [p.split("=")[1] for p in paths]
    return providers


def get_provider_countries(provider):
    """Get countries for a given provider."""
    import s3fs

    fs = s3fs.S3FileSystem(anon=True)
    glb = f"openaq-data-archive/records/csv.gz/provider={provider.lower()}/country=*"
    paths = fs.glob(glb, maxdepth=1)
    countries = [p.split("=")[2] for p in paths]
    return countries


def get_locations(*, provider=None, country=None):
    """Get location IDs corresponding to provider(s) and/or country(ies)."""
    import re

    import s3fs

    fs = s3fs.S3FileSystem(anon=True)
    country = _maybe_to_list(country)
    if provider is None:
        providers = get_providers()
    else:
        providers = _maybe_to_list(provider, not_none=True)

    paths = []
    for prvdr in providers:
        if country is None:
            countries = get_provider_countries(prvdr)
        else:
            countries = country

        for cntry in countries:
            glb = (
                "openaq-data-archive/records/csv.gz/"
                f"provider={prvdr.lower()}/country={cntry.lower()}/"
            )
            prvdr_cntry_paths = fs.find(glb, withdirs=True, maxdepth=1)
            paths.extend(prvdr_cntry_paths)

    rows = []
    for p in paths:
        m = re.fullmatch(
            r"openaq-data-archive/records/csv\.gz/"
            r"provider=([a-z0-9\-]+)/country=([a-z]{2}|\-\-|99|mobile)/"
            r"locationid=([0-9]+)",
            p,
        )
        if m is not None:
            rows.append(m.groups())

    df = pd.DataFrame(rows, columns=["provider", "country", "siteid"])
    return df


def build_urls(dates, sites, *, protocol="s3"):
    """Naively build URLs for OpenAQ archive data on AWS."""
    dates = _to_datetime_index(dates)
    sites = _maybe_to_list(sites, not_none=True)

    if protocol.lower() == "s3":
        pref = "s3://openaq-data-archive"
    elif protocol.lower() in {"http", "https"}:
        pref = f"{protocol.lower()}://openaq-data-archive.s3.amazonaws.com"
    else:
        raise ValueError(f"protocol: {protocol!r}")

    urls = []
    for site in sites:
        for date in dates.floor("D").unique():
            urls.append(
                f"{pref}/records/csv.gz/"
                f"locationid={site}/year={date:%Y}/month={date:%m}/"
                f"location-{site}-{date:%Y%m%d}.csv.gz"
            )

    return urls


# -----------------------------------------------------------------------------
# Legacy Compatibility
# -----------------------------------------------------------------------------


def read(fp, **kwargs):
    """Legacy wrapper."""
    return read_openaq_aws_csv(fp, **kwargs)


def _maybe_read(fp, **kwargs):
    """Legacy wrapper."""
    return read_openaq_aws_csv_robust(fp, **kwargs)


def _build_urls(dates, sites, **kwargs):
    """Legacy wrapper."""
    return build_urls(dates, sites, **kwargs)


def add_data(
    dates,
    *,
    siteid=None,
    country=None,
    provider=None,
    find_paths=True,
    wide_fmt=False,
    n_procs=1,
    **kwargs,
):
    """Helper for consistency."""
    return OpenAQAWSReader().open_dataset(
        dates=dates,
        siteid=siteid,
        country=country,
        provider=provider,
        find_paths=find_paths,
        wide_fmt=wide_fmt,
        n_procs=n_procs,
        as_xarray=False,  # Return DataFrame by default for add_data legacy
        **kwargs,
    )
