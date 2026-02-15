"""AQS Reader"""

import warnings
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    import dask.dataframe as dd

from monetio.obs.epa_util import read_monitor_file
from monetio.util import force_object_strings, long_to_wide

from .base import PointReader, register_reader


@register_reader("aqs")
class AQSReader(PointReader):
    def open_dataset(
        self,
        files: Optional[Union[str, List[str]]] = None,
        dates: Optional[Union[pd.DatetimeIndex, List[datetime], datetime, str]] = None,
        param: Optional[Union[str, List[str]]] = None,
        daily: bool = False,
        network: Optional[str] = None,
        download: bool = False,
        local: bool = False,
        wide_fmt: bool = True,
        n_procs: int = 1,
        meta: bool = False,
        as_xarray: bool = True,
        lazy: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, xr.Dataset, "dd.DataFrame"]:
        """
        Retrieve and load AQS (Air Quality System) data.

        Parameters
        ----------
        files : Union[str, List[str]], optional
            File path, list of paths, or glob pattern.
        dates : Union[pd.DatetimeIndex, List[datetime], datetime, str], optional
            Dates to retrieve if files are not provided.
        param : Union[str, List[str]], optional
            Parameter(s) to retrieve (e.g., 'OZONE', 'PM2.5'), by default None.
        daily : bool, optional
            Whether to load daily data, by default False.
        network : str, optional
            Network to filter sites, by default None.
        download : bool, optional
            Whether to download files, by default False.
        local : bool, optional
            Whether to load from local files, by default False.
        wide_fmt : bool, optional
            Whether to return data in wide format, by default True.
        n_procs : int, optional
            Number of processors for dask compute, by default 1.
        meta : bool, optional
            Whether to add site metadata, by default False.
        as_xarray : bool, optional
            Whether to return an xarray.Dataset, by default True.
        lazy : bool, optional
            Whether to return a dask-backed object, by default False.
        **kwargs : dict
            Additional arguments passed to the reader.

        Returns
        -------
        Union[pd.DataFrame, xr.Dataset, dd.DataFrame]
            The loaded AQS data.
        """
        a = AQS()

        if files is None:
            if dates is None:
                raise ValueError("Must provide either 'files' or 'dates'.")

            # Build URLs
            params = a._get_param_list(param)
            urls, fnames = a.build_urls(params, dates, daily=daily)

            if not urls:
                return pd.DataFrame()

            if download:
                for url, fname in zip(urls, fnames):
                    a.retrieve(url, fname)
                files = fnames
            elif local:
                files = fnames
            else:
                files = urls

        # Use PandasDriver via base class
        # Pass a partial of load_aqs_file as the read_method
        read_func = partial(a.load_aqs_file, network=network)

        df = super().open_dataset(
            files,
            read_method=read_func,
            as_xarray=False,
            lazy=lazy,
            **kwargs,
        )

        if len(df) == 0:
            return df

        # Filter dates
        if dates is not None:
            dates_idx = pd.DatetimeIndex(np.atleast_1d(pd.to_datetime(dates)))
            # Backend agnostic filter
            df = df[df.time.between(dates_idx.min(), dates_idx.max())]

        if meta:
            df = a.add_metadata(df, daily=daily, network=network)

        # We only perform wide_fmt here if NOT lazy, to avoid the hidden compute in long_to_wide
        do_wide = wide_fmt and not lazy
        if do_wide:
            df = long_to_wide(df)

        df = self.harmonize(df)

        if not lazy and hasattr(df, "compute"):
            df = df.compute(num_workers=n_procs)

        if as_xarray:
            ds = self.to_xarray(df)
            # Update history
            history = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read AQS data."
            if "history" in ds.attrs:
                ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
            else:
                ds.attrs["history"] = history

            if wide_fmt and lazy and "variable" in ds.data_vars:
                warnings.warn(
                    "AQS: Dataset is in 'long' format because lazy=True. "
                    "Use ds.to_dataset(dim='variable') or similar to pivot lazily.",
                    UserWarning,
                )

            return ds

        return df


class AQS:
    """Helper class for AQS data retrieval and processing."""

    def __init__(self):
        self.baseurl = "https://aqs.epa.gov/aqsweb/airdata/"
        self.renameddcols = [
            "time",
            "state_code",
            "county_code",
            "site_num",
            "parameter_code",
            "poc",
            "latitude",
            "longitude",
            "datum",
            "parameter_name",
            "sample_duration",
            "pollutant_standard",
            "units",
            "event_type",
            "observation_count",
            "observation_percent",
            "obs",
            "1st_max_value",
            "1st_max_hour",
            "aqi",
            "method_code",
            "method_name",
            "local_site_name",
            "address",
            "state_name",
            "county_name",
            "city_name",
            "msa_name",
            "date_of_last_change",
        ]

    def _get_param_list(self, param: Optional[Union[str, List[str]]]) -> List[str]:
        if param is None:
            return [
                "SPEC",
                "PM10",
                "PM2.5",
                "PM2.5_FRM",
                "CO",
                "OZONE",
                "SO2",
                "VOC",
                "NONOXNOY",
                "WIND",
                "TEMP",
                "RHDP",
            ]
        elif isinstance(param, str):
            return [param]
        return param

    def columns_rename(self, columns: List[str], verbose: bool = False) -> List[str]:
        """Rename AQS columns to standard names."""
        rcolumn = []
        for ccc in columns:
            ccc_clean = ccc.strip()
            if ccc_clean == "Sample Measurement":
                newc = "obs"
            elif ccc_clean == "Units of Measure":
                newc = "units"
            else:
                newc = ccc_clean.lower().replace(" ", "_")
            if verbose:
                print(f"{ccc} renamed {newc}")
            rcolumn.append(newc)
        return rcolumn

    def load_aqs_file(self, url: str, network: Optional[str] = None) -> pd.DataFrame:
        """Load a single AQS file."""
        if "daily" in url:
            df = pd.read_csv(
                url,
                dtype={0: str, 1: str, 2: str},
                encoding="ISO-8859-1",
            )
            # Find column for time_local
            if "Date Local" in df.columns:
                df["time_local"] = pd.to_datetime(df["Date Local"])
                df.drop(["Date Local"], axis=1, inplace=True)

            # Reorder columns to match renameddcols (first column is time_local)
            cols = df.columns.tolist()
            if "time_local" in cols:
                cols.insert(0, cols.pop(cols.index("time_local")))
                df = df[cols]

            if len(df.columns) == len(self.renameddcols):
                df.columns = self.renameddcols

            df["pollutant_standard"] = df.get("pollutant_standard", pd.Series(dtype=str)).astype(
                str
            )
        else:
            df = pd.read_csv(
                url,
                low_memory=False,
            )
            # Handle dates manually
            if "Date GMT" in df.columns and "Time GMT" in df.columns:
                df["time"] = pd.to_datetime(df["Date GMT"] + " " + df["Time GMT"])
            if "Date Local" in df.columns and "Time Local" in df.columns:
                df["time_local"] = pd.to_datetime(df["Date Local"] + " " + df["Time Local"])

            df.columns = self.columns_rename(df.columns.values)
            # Remove duplicate time_local if it was created from 'Time Local'
            df = df.loc[:, ~df.columns.duplicated()]

        df["siteid"] = (
            df.state_code.astype(str).str.zfill(2)
            + df.county_code.astype(str).str.zfill(3)
            + df.site_num.astype(str).str.zfill(4)
        )
        df.drop(["state_name", "county_name"], axis=1, inplace=True, errors="ignore")
        df.columns = [i.lower() for i in df.columns]
        if "daily" not in url:
            df.drop(["datum", "qualifier"], axis=1, inplace=True, errors="ignore")
        voc = "VOC" in url
        df = self.get_species(df, voc=voc)
        df = self.change_units(df)
        df = force_object_strings(df)
        return df.drop(columns="date_of_last_change", errors="ignore")

    def build_url(self, param: str, year: str, daily: bool = False) -> tuple:
        """Build URL and filename for a given parameter and year."""
        if daily:
            beginning = self.baseurl + "daily_"
            fname_prefix = "daily_"
        else:
            beginning = self.baseurl + "hourly_"
            fname_prefix = "hourly_"

        p = param.upper()
        mapping = {
            "OZONE": "44201_",
            "O3": "44201_",
            "PM2.5": "88101_",
            "PM2.5_FRM": "88502_",
            "PM10": "81102_",
            "SO2": "42401_",
            "NO2": "42602_",
            "CO": "42101_",
            "NONOXNOY": "NONOxNOy_",
            "VOC": "VOCS_",
            "SPEC": "SPEC_",
            "PM10SPEC": "PM10SPEC_",
            "WIND": "WIND_",
            "TEMP": "TEMP_",
            "RHDP": "RH_DP_",
            "WS": "WIND_",
            "WDIR": "WIND_",
        }
        code = mapping.get(p, p + "_")

        url = f"{beginning}{code}{year}.zip"
        fname = f"{fname_prefix}{code}{year}.zip"
        return url, fname

    def build_urls(self, params: List[str], dates, daily: bool = False) -> tuple:
        """Build multiple URLs for given parameters and dates."""
        import requests

        years = pd.DatetimeIndex(np.atleast_1d(pd.to_datetime(dates))).year.unique().astype(str)
        urls = []
        fnames = []
        for i in params:
            for y in years:
                url, fname = self.build_url(i, y, daily=daily)
                try:
                    with requests.get(url, stream=True, timeout=10) as r:
                        if r.status_code == 200:
                            content_length = int(r.headers.get("Content-Length", 0))
                            if content_length > 500:
                                urls.append(url)
                                fnames.append(fname)
                            else:
                                print(f"File is Empty. Not Processing {url}")
                except Exception:
                    pass
        return urls, fnames

    def retrieve(self, url: str, fname: str):
        """Retrieve a file from a URL."""
        import os

        import requests

        if not os.path.isfile(fname):
            print(f"\n Retrieving: {fname}")
            print(url)
            r = requests.get(url)
            with open(fname, "wb") as f:
                f.write(r.content)
        else:
            print(f"\n File Exists: {fname}")

    def add_metadata(
        self, df: Union[pd.DataFrame, "dd.DataFrame"], daily: bool = False, network: str = None
    ) -> Union[pd.DataFrame, "dd.DataFrame"]:
        """Add site metadata and adjust time for daily data."""
        try:
            import dask.dataframe as dd

            is_dask = isinstance(df, dd.DataFrame)
        except ImportError:
            is_dask = False

        monitor_df = read_monitor_file()

        if network is not None:
            monitors = monitor_df.loc[monitor_df.isin([network]).any(axis=1)].drop_duplicates(
                subset=["siteid"]
            )
        else:
            monitors = monitor_df.drop_duplicates(subset=["siteid"])

        # Ensure siteid is object for reliable merging
        monitors = monitors.copy()
        monitors["siteid"] = monitors["siteid"].astype(object)

        if is_dask:
            df["siteid"] = df["siteid"].astype(object)
            monitors_dask = dd.from_pandas(monitors, npartitions=1)
            df = df.merge(monitors_dask, on="siteid", how="left")
        else:
            df["siteid"] = df["siteid"].astype(object)
            df = df.merge(monitors, on="siteid", how="left")

        if daily and "gmt_offset" in df.columns:
            # Adjust time for daily data based on local time and offset
            if is_dask:
                df["time"] = df.time_local - dd.to_timedelta(df.gmt_offset, unit="h")
            else:
                df["time"] = df.time_local - pd.to_timedelta(df.gmt_offset, unit="h")

        if "parameter_name" in df.columns:
            df = df.drop(columns="parameter_name")

        return df

    def get_species(
        self, df: Union[pd.DataFrame, "dd.DataFrame"], voc: bool = False
    ) -> Union[pd.DataFrame, "dd.DataFrame"]:
        """Map parameter codes to short variable names."""
        if voc:
            df["variable"] = df.parameter_name.str.upper()
            return df

        mapping = {
            88101: "PM2.5",
            88502: "PM2.5",
            44201: "OZONE",
            81102: "PM10",
            42401: "SO2",
            42602: "NO2",
            42101: "CO",
            62101: "TEMP",
            88305: "OC",
            88306: "NO3f",
            88307: "ECf",
            88316: "ECf_optical",
            88403: "SO4f",
            88312: "TCf",
            88104: "Alf",
            88107: "Baf",
            88313: "BCf",
            88109: "Brf",
            88110: "Cdf",
            88111: "Caf",
            88117: "Cef",
            88118: "Csf",
            88203: "Cl-f",
            88115: "Clf",
            88112: "Crf",
            88113: "Cof",
            88114: "Cuf",
            88121: "Euf",
            88143: "Auf",
            88127: "Hff",
            88131: "Inf",
            88126: "Fef",
            88146: "Laf",
            88128: "Pbf",
            88140: "Mgf",
            88132: "Mnf",
            88142: "Hgf",
            88134: "Mof",
            88136: "Nif",
            88147: "Nbf",
            88310: "NO3f",
            88152: "Pf",
            88303: "K+f",
            88176: "Rbf",
            88162: "Smf",
            88163: "Scf",
            88154: "Sef",
            88165: "Sif",
            88166: "Agf",
            88302: "Na+f",
            88184: "Naf",
            88168: "Srf",
            88169: "Sf",
            88170: "Taf",
            88172: "Tbf",
            88160: "Snf",
            88161: "Tif",
            88186: "Wf",
            88314: "C_370nmf",
            88179: "Uf",
            88164: "Vf",
            88183: "Yf",
            88167: "Znf",
            88185: "Zrf",
            88103: "Asf",
            88105: "Bef",
            88124: "Gaf",
            88180: "Kf",
            88301: "NH4+f",
            42600: "NOY",
            42601: "NO",
            42603: "NOX",
            61103: "WS",
            61101: "WS",
            61104: "WD",
            61102: "WD",
            62201: "RH",
            62103: "DP",
        }

        # Convert keys to string for matching
        mapping_str = {str(k): v for k, v in mapping.items()}

        if "variable" not in df.columns:
            df["variable"] = ""

        pcode_as_str = df["parameter_code"].astype(str)
        df["variable"] = pcode_as_str.map(mapping_str)

        # Handle missing mappings
        if "variable" in df.columns:
            # For the warning, we check if any are missing.
            # To stay lazy, we only do this if df is not a dask dataframe.
            if not hasattr(df, "compute"):
                missing = df.loc[
                    df["variable"].isna(), ["parameter_name", "parameter_code"]
                ].drop_duplicates()
                if not missing.empty:
                    _tbl = missing.to_string(index=False)
                    warnings.warn(f"Short names not available for these variables:\n{_tbl}")

        df["variable"] = df["variable"].fillna(df["parameter_name"])

        return df

    @staticmethod
    def change_units(
        df: Union[pd.DataFrame, "dd.DataFrame"],
    ) -> Union[pd.DataFrame, "dd.DataFrame"]:
        """Standardize units and adjust observation values accordingly."""
        # Knots to m/s
        is_knots = df.units.str.lower() == "knots"
        df["obs"] = df["obs"].mask(is_knots, df.obs * 0.51444)
        df["units"] = df["units"].mask(is_knots, "m/s")

        # Fahrenheit to Kelvin
        is_f = df.units.str.lower() == "degrees fahrenheit"
        df["obs"] = df["obs"].mask(is_f, (df.obs + 459.67) * 5.0 / 9.0)
        df["units"] = df["units"].mask(is_f, "k")

        # Others (just rename)
        unit_map = {
            "parts per billion carbon": "ppbC",
            "parts per billion": "ppb",
            "parts per million": "ppm",
            "micrograms/cubic meter (25 c)": "ug/m3",
            "micrograms/cubic meter (lc)": "ug/m3",
            "degrees centigrade": "c",
            "percent relative humidity": "%",
        }

        # Apply mapping to units column
        df["units_lower"] = df.units.str.lower()
        for old, new in unit_map.items():
            df["units"] = df["units"].mask(df.units_lower == old, new)

        df = df.drop(columns="units_lower")

        return df
