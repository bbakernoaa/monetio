"""OpenAQ Reader"""

import json

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd

from .base import PointReader, register_reader


@register_reader("openaq")
class OpenAQReader(PointReader):
    def open_dataset(self, dates, n_procs=1, wide_fmt=True, **kwargs):
        """
        Reads OpenAQ data from S3.
        """
        a = OPENAQ()
        return a.add_data(dates, num_workers=n_procs, wide_fmt=wide_fmt)


# -----------------------------------------------------------------------------
# Helper functions ported from monetio/obs/openaq.py
# -----------------------------------------------------------------------------


def read_json(fp_or_url, verbose=False):

    # Simple pandas read if local or url
    try:
        df = pd.read_json(fp_or_url, lines=True)
    except Exception as e:
        # Fallback if S3 URL is not directly readable by pandas without s3fs installed/configured
        # But pandas supports s3:// if s3fs is present.
        raise e

    if "attribution" in df.columns:
        df = df.drop(columns="attribution")

    df = df.dropna(subset=["coordinates"])
    to_expand = ["date", "averagingPeriod", "coordinates"]
    # Check if columns exist
    to_expand = [c for c in to_expand if c in df.columns]

    if not to_expand:
        return df

    new = pd.json_normalize(json.loads(df[to_expand].to_json(orient="records")))

    if "date.utc" in new.columns:
        time = pd.to_datetime(new["date.utc"]).dt.tz_localize(None)
    else:
        time = pd.Series(index=new.index, dtype="datetime64[ns]")

    if "date.local" in new.columns:
        # Simple offset parsing: -06:00 -> -6h
        # This is fragile but matches original code intent
        try:
            utcoffset = pd.to_timedelta(new["date.local"].str.slice(-6, None) + ":00")
        except Exception:
            utcoffset = pd.Timedelta(0)
    else:
        utcoffset = pd.Timedelta(0)

    time_local = time + utcoffset

    # Averaging period
    averagingPeriod = pd.Series(np.full(len(new), np.nan, dtype="timedelta64[ns]"))
    if "averagingPeriod.value" in new.columns and "averagingPeriod.unit" in new.columns:
        value = new["averagingPeriod.value"]
        units = new["averagingPeriod.unit"]
        unique_units = units.dropna().unique()
        for unit in unique_units:
            is_unit = units == unit
            averagingPeriod.loc[is_unit] = pd.to_timedelta(value[is_unit], unit=unit)

    # Apply new columns
    df = df.drop(columns=to_expand).assign(
        time=time,
        time_local=time_local,
        utcoffset=utcoffset,
        averagingPeriod=averagingPeriod,
    )
    if "coordinates.latitude" in new.columns:
        df["latitude"] = new["coordinates.latitude"]
    if "coordinates.longitude" in new.columns:
        df["longitude"] = new["coordinates.longitude"]

    return df


class OPENAQ:
    NON_MOLEC_PARAMS = ["pm1", "pm25", "pm4", "pm10", "bc"]
    PPM_TO_UGM3 = {
        "o3": 1990,
        "co": 1160,
        "no2": 1900,
        "no": 1240,
        "so2": 2650,
        "ch4": 664,
        "co2": 1820,
    }
    # NOx assumption
    PPM_TO_UGM3["nox"] = PPM_TO_UGM3["no2"]

    def __init__(self):
        import s3fs

        self.fs = s3fs.S3FileSystem(anon=True)
        self.s3bucket = "openaq-fetches/realtime"
        # We use pandas engine by default
        self.read = read_json

    def _get_available_days(self, dates):
        # Listing S3 is slow, mocking if possible or optimizing
        # Original code lists all folders.
        # We assume dates requested are valid to speed up or rely on try/catch
        # But to match original logic:
        try:
            folders = self.fs.ls(self.s3bucket)
            days = [folder.split("/")[2] for folder in folders]
            dates_available = pd.Series(
                pd.to_datetime(days, format=r"%Y-%m-%d", errors="coerce"), name="dates"
            )
            dates_requested = pd.Series(
                pd.to_datetime(dates).floor(freq="D"), name="dates"
            ).drop_duplicates()
            dates_have = pd.merge(dates_available, dates_requested, how="inner")["dates"]
            if dates_have.empty:
                raise ValueError(f"No data available for requested dates: {dates_requested}.")
            return dates_have
        except Exception:
            # If fs.ls fails (no internet), return requested dates assuming they exist?
            # Or raise.
            raise

    def _get_files_in_day(self, date):
        sdate = date.strftime(r"%Y-%m-%d")
        try:
            files = self.fs.ls(f"{self.s3bucket}/{sdate}")
            return files
        except Exception:
            return []

    def build_urls(self, dates):
        dates_ = self._get_available_days(dates)
        urls = []
        for date in dates_:
            files = self._get_files_in_day(date)
            urls.extend(f"s3://{f}" for f in files)
        return urls

    def add_data(self, dates, *, num_workers=1, wide_fmt=True):
        dates = pd.to_datetime(dates)
        if isinstance(dates, pd.Timestamp):
            dates = pd.DatetimeIndex([dates])
        dates = dates.sort_values()

        urls = self.build_urls(dates)

        dfs = [dask.delayed(self.read)(url) for url in urls]
        if not dfs:
            return pd.DataFrame()

        df_lazy = dd.from_delayed(dfs)
        df = df_lazy.compute(num_workers=num_workers)

        df = df.loc[(df.time >= dates.min()) & (df.time <= dates.max())]

        # Convert units
        if wide_fmt:
            for vn, f in self.PPM_TO_UGM3.items():
                is_ug = (df.parameter == vn) & (df.unit == "µg/m³")
                df.loc[is_ug, "value"] /= f
                df.loc[is_ug, "unit"] = "ppm"

            index = [
                "time",
                "time_local",
                "latitude",
                "longitude",
                "utcoffset",
                "location",
                "city",
                "country",
                "sourceName",
                "sourceType",
                "mobile",
            ]
            # remove index cols that might not exist
            index = [c for c in index if c in df.columns]

            df = df.pivot_table(values="value", index=index, columns="parameter").reset_index()

            # Renames
            df = df.rename(columns={p: f"{p}_ugm3" for p in self.NON_MOLEC_PARAMS}, errors="ignore")
            df = df.rename(columns={p: f"{p}_ppm" for p in self.PPM_TO_UGM3}, errors="ignore")

        # Site ID hash
        if "location" in df.columns and "country" in df.columns:
            import hashlib

            def do_hash(b):
                return hashlib.sha1(b.encode("utf-8")).hexdigest()

            to_hash = (
                df.location.astype(str)
                + " "
                + df.latitude.astype(str)
                + " "
                + df.longitude.astype(str)
            )
            df["siteid"] = df.country + "_" + to_hash.apply(do_hash).str.slice(0, 7)

        return df
