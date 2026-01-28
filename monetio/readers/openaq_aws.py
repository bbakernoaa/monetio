"""OpenAQ archive data on AWS.

https://openaq.org/
https://registry.opendata.aws/openaq/
https://docs.openaq.org/aws/about
"""

import logging
import warnings

import pandas as pd

from .base import PointReader, register_reader

logger = logging.getLogger(__name__)


def read(fp):
    """Read OpenAQ archive data from a file-like object.

    Parameters
    ----------
    fp : str or path-like or file-like
        OpenAQ archive data, suitable for passing to ``pd.read_csv``.

    Returns
    -------
    pd.DataFrame
        OpenAQ archive data.
    """

    df = pd.read_csv(
        fp,
        dtype={
            0: str,  # location_id
            1: str,  # sensor_id or sensors_id ??
            2: str,  # location
            3: str,  # datetime
            4: float,  # lat
            5: float,  # lon
            6: str,  # parameter
            7: str,  # unit or units ??
            8: float,  # value
        },
        parse_dates=["datetime"],
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
        df["time"] = df["time"].dt.tz_convert("UTC").dt.tz_localize(None)

    return df


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


def _build_urls(dates, sites, *, protocol="s3"):
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


def _maybe_read(fp):
    """Try to read a file, returning empty DF if not found."""
    try:
        return read(fp)
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


@register_reader("openaq_aws")
class OpenAQAWSReader(PointReader):
    """OpenAQ AWS Archive Reader"""

    def open_dataset(
        self,
        dates,
        *,
        siteid=None,
        country=None,
        provider=None,
        find_paths=True,
        n_procs=1,
        **kwargs,
    ):
        """Add OpenAQ data from AWS Open Data."""
        import dask.dataframe as dd

        dates = _to_datetime_index(dates).dropna()
        if dates.empty:
            raise ValueError("must provide at least one datetime-like")

        if find_paths:
            paths = get_paths(dates, siteid=siteid, country=country, provider=provider)
            urls = [f"s3://{p}" for p in paths]
            func = read
        else:
            if siteid is None:
                raise ValueError("must provide `siteid` when `find_paths` is false")
            urls = _build_urls(dates, siteid)
            func = _maybe_read

        meta = [
            ("siteid", str),
            ("sensor_id", str),
            ("location", str),
            ("time", "datetime64[ns]"),
            ("latitude", float),
            ("longitude", float),
            ("parameter", str),
            ("units", str),
            ("value", float),
        ]
        df = dd.from_map(func, urls, meta=meta).compute(num_workers=n_procs)

        ds = df.reset_index(drop=True)
        ds.attrs["history"] = (
            f"Read OpenAQ AWS Archive data for dates {dates.min()} to {dates.max()}"
        )
        return ds


def add_data(
    dates,
    *,
    siteid=None,
    country=None,
    provider=None,
    find_paths=True,
    n_procs=1,
):
    """Helper for Aero Protocol consistency."""
    return OpenAQAWSReader().open_dataset(
        dates=dates,
        siteid=siteid,
        country=country,
        provider=provider,
        find_paths=find_paths,
        n_procs=n_procs,
    )
