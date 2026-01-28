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
    # Convert HTTP S3 URLs to s3:// to avoid 403 Forbidden issues in some environments
    # but only if we want to use s3fs.
    # Actually, let's stick to original behavior as much as possible for test compatibility.

    # Simple pandas read if local or url
    try:
        df = pd.read_json(fp_or_url, lines=True)
    except Exception as e:
        # If it's an S3 URL and failed, try s3fs
        if isinstance(fp_or_url, str) and (
            fp_or_url.startswith("s3://") or "s3.amazonaws.com" in fp_or_url
        ):
            try:
                import s3fs

                # Convert to s3:// if it's HTTP
                s3_path = fp_or_url
                if "openaq-fetches.s3.amazonaws.com" in s3_path:
                    s3_path = s3_path.replace(
                        "https://openaq-fetches.s3.amazonaws.com", "s3://openaq-fetches"
                    )
                    s3_path = s3_path.replace(
                        "http://openaq-fetches.s3.amazonaws.com", "s3://openaq-fetches"
                    )

                fs = s3fs.S3FileSystem(anon=True)
                with fs.open(s3_path, "rb") as f:
                    df = pd.read_json(f, lines=True)
            except Exception:
                raise e
        else:
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
        try:
            # Handle possible varied offset formats
            utcoffset = pd.to_timedelta(
                new["date.local"]
                .str.slice(-6, None)
                .str.replace(r"(\d{2})(\d{2})$", r"\1:\2", regex=True)
            )
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
            # unit might be 'hours' etc.
            try:
                averagingPeriod.loc[is_unit] = pd.to_timedelta(value[is_unit], unit=unit)
            except Exception:
                pass

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


def read_json2(fp_or_url, verbose=False):
    import datetime

    import requests

    if isinstance(fp_or_url, str) and fp_or_url.startswith("s3"):
        fp_or_url = fp_or_url.replace(
            "s3://openaq-fetches/", "https://openaq-fetches.s3.amazonaws.com/"
        )

    r = requests.get(fp_or_url, stream=True, timeout=10)
    r.raise_for_status()

    names = [
        "time",
        "utcoffset",
        "latitude",
        "longitude",
        "parameter",
        "value",
        "unit",
        "averagingPeriod",
        "location",
        "city",
        "country",
        "attribution",
        "sourceName",
        "sourceType",
        "mobile",
    ]
    rows = []
    for line in r.iter_lines():
        if line:
            data = json.loads(line)
            coords = data.get("coordinates")
            if coords is None:
                continue

            # Time
            try:
                time = datetime.datetime.fromisoformat(
                    data["date"]["utc"].replace("Z", "+00:00")
                ).replace(tzinfo=None)
                time_local_str = data["date"]["local"]
                # -06:00
                h = int(time_local_str[-6:-3])
                m = int(time_local_str[-2:])
                utcoffset = datetime.timedelta(hours=h, minutes=m)
            except Exception:
                time = datetime.datetime(1970, 1, 1)
                utcoffset = datetime.timedelta(0)

            # Averaging period
            ap = data.get("averagingPeriod")
            if ap is not None:
                val = data["averagingPeriod"]["value"]
                unit = data["averagingPeriod"]["unit"]
                # Unit might need mapping for timedelta
                try:
                    averagingPeriod = datetime.timedelta(**{unit: val})
                except Exception:
                    averagingPeriod = None
            else:
                averagingPeriod = None

            # Attribution
            attrs = data.get("attribution")
            attr_name = attrs[0]["name"] if attrs else None

            rows.append(
                (
                    time,
                    utcoffset,
                    data["coordinates"]["latitude"],
                    data["coordinates"]["longitude"],
                    data["parameter"],
                    data["value"],
                    data["unit"],
                    averagingPeriod,
                    data["location"],
                    data["city"],
                    data["country"],
                    attr_name,
                    data["sourceName"],
                    data["sourceType"],
                    data["mobile"],
                )
            )

    df = pd.DataFrame(rows, columns=names)
    df["time_local"] = df["time"] + df["utcoffset"]
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
    PPM_TO_UGM3["nox"] = PPM_TO_UGM3["no2"]

    def __init__(self, engine="pandas"):
        import s3fs

        self.fs = s3fs.S3FileSystem(anon=True)
        self.s3bucket = "openaq-fetches/realtime"
        self.engine = engine
        if engine == "pandas":
            self.read = read_json
        else:
            self.read = read_json2

    def _get_available_days(self, dates):
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

    def _get_files_in_day(self, date):
        sdate = date.strftime(r"%Y-%m-%d")
        return self.fs.ls(f"{self.s3bucket}/{sdate}")

    def build_urls(self, dates):
        dates_ = self._get_available_days(dates)
        urls = []
        for date in dates_:
            files = self._get_files_in_day(date)
            urls.extend(f"s3://{f}" for f in files)
        return urls

    def add_data(self, dates, *, num_workers=1, wide_fmt=True):
        import hashlib

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

        # SITE ID
        def do_hash(b):
            return hashlib.sha1(b).hexdigest()

        to_hash = df.location + " " + df.latitude.astype(str) + " " + df.longitude.astype(str)
        df["siteid"] = df.country + "_" + to_hash.str.encode("utf-8").apply(do_hash).str.slice(0, 7)

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
                "siteid",
            ]
            if self.engine != "pandas":
                index.append("attribution")

            index = [c for c in index if c in df.columns]

            df = df.pivot_table(values="value", index=index, columns="parameter").reset_index()
            df = df.rename(columns={p: f"{p}_ugm3" for p in self.NON_MOLEC_PARAMS}, errors="ignore")
            df = df.rename(columns={p: f"{p}_ppm" for p in self.PPM_TO_UGM3}, errors="ignore")

        return df
