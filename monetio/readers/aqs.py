"""AQS Reader"""

import os
import warnings

import pandas as pd

from monetio.obs.epa_util import read_monitor_file
from monetio.util import long_to_wide

from .base import PointReader, register_reader


@register_reader("aqs")
class AQSReader(PointReader):
    def open_dataset(
        self,
        dates,
        param=None,
        daily=False,
        network=None,
        download=False,
        local=False,
        wide_fmt=True,
        n_procs=1,
        meta=False,
        as_xarray=True,
        **kwargs,
    ):
        """
        Reads AQS data.
        """
        a = AQS()
        df = a.add_data(
            dates,
            param=param,
            daily=daily,
            network=network,
            download=download,
            local=local,
            n_procs=n_procs,
            meta=meta,
        )

        if wide_fmt:
            df = long_to_wide(df)

        df = self.harmonize(df)
        if as_xarray:
            return self.to_xarray(df)

        return df


# -----------------------------------------------------------------------------
# Helper functions ported from monetio/obs/aqs.py
# -----------------------------------------------------------------------------


class AQS:
    def __init__(self):
        self.objtype = "AQS"
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
        self.savecols = [
            "time_local",
            "time",
            "siteid",
            "latitude",
            "longitude",
            "obs",
            "units",
            "variable",
        ]
        self.df = pd.DataFrame()
        self.monitor_file = None
        self.monitor_df = None
        self.daily = False
        self.d_df = None

    def columns_rename(self, columns, verbose=False):
        rcolumn = []
        for ccc in columns:
            if ccc.strip() == "Sample Measurement":
                newc = "obs"
            elif ccc.strip() == "Units of Measure":
                newc = "units"
            else:
                newc = ccc.strip().lower().replace(" ", "_")
            if verbose:
                print(ccc + " renamed " + newc)
            rcolumn.append(newc)
        return rcolumn

    def load_aqs_file(self, url, network):
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

            # Reorder columns to match renameddcols (first column is time_local, now named time in the result)
            cols = df.columns.tolist()
            if "time_local" in cols:
                cols.insert(0, cols.pop(cols.index("time_local")))
                df = df[cols]

            if len(df.columns) == len(self.renameddcols):
                df.columns = self.renameddcols

            df["pollutant_standard"] = df.get("pollutant_standard", pd.Series(dtype=str)).astype(
                str
            )
            self.daily = True
        else:
            df = pd.read_csv(
                url,
                low_memory=False,
            )
            # Handle dates manually to avoid deprecated parse_dates
            if "Date GMT" in df.columns and "Time GMT" in df.columns:
                df["time"] = pd.to_datetime(df["Date GMT"] + " " + df["Time GMT"])
            if "Date Local" in df.columns and "Time Local" in df.columns:
                df["time_local"] = pd.to_datetime(df["Date Local"] + " " + df["Time Local"])

            df.columns = self.columns_rename(df.columns.values)

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
        return df.drop("date_of_last_change", axis=1, errors="ignore")

    def build_url(self, param, year, daily=False, download=False):
        if daily:
            beginning = self.baseurl + "daily_"
            fname = "daily_"
        else:
            beginning = self.baseurl + "hourly_"
            fname = "hourly_"

        p = param.upper()
        if p in ["OZONE", "O3"]:
            code = "44201_"
        elif p == "PM2.5":
            code = "88101_"
        elif p == "PM2.5_FRM":
            code = "88502_"
        elif p == "PM10":
            code = "81102_"
        elif p == "SO2":
            code = "42401_"
        elif p == "NO2":
            code = "42602_"
        elif p == "CO":
            code = "42101_"
        elif p == "NONOXNOY":
            code = "NONOxNOy_"
        elif p == "VOC":
            code = "VOCS_"
        elif p == "SPEC":
            code = "SPEC_"
        elif p == "PM10SPEC":
            code = "PM10SPEC_"
        elif p == "WIND":
            code = "WIND_"
        elif p == "TEMP":
            code = "TEMP_"
        elif p == "RHDP":
            code = "RH_DP_"
        elif p in ["WS", "WDIR"]:
            code = "WIND_"
        else:
            code = p + "_"

        url = beginning + code + year + ".zip"
        fname = fname + code + year + ".zip"
        return url, fname

    def build_urls(self, params, dates, daily=False):
        import requests

        years = pd.DatetimeIndex(dates).year.unique().astype(str)
        urls = []
        fnames = []
        for i in params:
            for y in years:
                url, fname = self.build_url(i, y, daily=daily)
                try:
                    # Using stream=True and Content-Length check to avoid downloading big files just for check
                    with requests.get(url, stream=True, timeout=10) as r:
                        if r.status_code == 200:
                            content_length = int(r.headers.get("Content-Length", 0))
                            if content_length > 500:
                                urls.append(url)
                                fnames.append(fname)
                            else:
                                print("File is Empty. Not Processing", url)
                except Exception:
                    pass
        return urls, fnames

    def retrieve(self, url, fname):
        import requests

        if not os.path.isfile(fname):
            print("\n Retrieving: " + fname)
            print(url)
            r = requests.get(url)
            with open(fname, "wb") as f:
                f.write(r.content)
        else:
            print("\n File Exists: " + fname)

    def add_data(
        self,
        dates,
        param=None,
        daily=False,
        network=None,
        download=False,
        local=False,
        n_procs=1,
        meta=False,
    ):
        import dask
        import dask.dataframe as dd

        dates = pd.DatetimeIndex(dates)

        if param is None:
            params = [
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
            params = [param]
        else:
            params = param

        urls, fnames = self.build_urls(params, dates, daily=daily)

        if download:
            for url, fname in zip(urls, fnames):
                self.retrieve(url, fname)
            dfs = [dask.delayed(self.load_aqs_file)(i, network) for i in fnames]
        elif local:
            dfs = [dask.delayed(self.load_aqs_file)(i, network) for i in fnames]
        else:
            dfs = [dask.delayed(self.load_aqs_file)(i, network) for i in urls]

        if not dfs:
            return pd.DataFrame()

        dff = dd.from_delayed(dfs)
        dfff = dff.compute(num_workers=n_procs)
        dfff = dfff[dfff.time.between(dates.min(), dates.max())]

        if meta:
            return self.add_data2(dfff, daily, network)
        else:
            return dfff

    def add_data2(self, df, daily=False, network=None):
        self.df = df
        self.df = self.change_units(self.df)
        if self.monitor_df is None:
            self.monitor_df = read_monitor_file()

        if network is not None:
            monitors = self.monitor_df.loc[
                self.monitor_df.isin([network]).any(axis=1)
            ].drop_duplicates(subset=["siteid"])
        else:
            monitors = self.monitor_df.drop_duplicates(subset=["siteid"])

        mlist = ["siteid"]
        self.df = pd.merge(self.df, monitors, on=mlist, how="left")

        if daily and "gmt_offset" in self.df.columns:
            self.df["time"] = self.df.time_local - pd.to_timedelta(self.df.gmt_offset, unit="h")

        if "parameter_name" in self.df.columns:
            self.df.drop("parameter_name", axis=1, inplace=True)

        return self.df

    def get_species(self, df, voc=False):
        pc = df.parameter_code.unique()
        df["variable"] = ""
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

        for i in pc:
            con = df.parameter_code == i
            try:
                # Try both int and string match
                val = mapping.get(i) or mapping.get(int(i))
                if val:
                    df.loc[con, "variable"] = val
            except Exception:
                pass

        con = df.variable == ""
        if con.sum() > 0:
            _tbl = (
                df[con][["parameter_name", "parameter_code"]]
                .drop_duplicates("parameter_name")
                .to_string(index=False)
            )
            warnings.warn(f"Short names not available for these variables:\n{_tbl}")
            df.loc[con, "variable"] = df.parameter_name

        return df

    @staticmethod
    def change_units(df):
        units = df.units.unique()
        for i in units:
            con = df.units == i
            if i.upper() == "Parts per billion Carbon".upper():
                df.loc[con, "units"] = "ppbC"
            if i == "Parts per billion":
                df.loc[con, "units"] = "ppb"
            if i == "Parts per million":
                df.loc[con, "units"] = "ppm"
            if i == "Micrograms/cubic meter (25 C)":
                df.loc[con, "units"] = "UG/M3".lower()
            if i == "Degrees Centigrade":
                df.loc[con, "units"] = "C"
            if i == "Micrograms/cubic meter (LC)":
                df.loc[con, "units"] = "UG/M3".lower()
            if i == "Knots":
                df.loc[con, "obs"] *= 0.51444
                df.loc[con, "units"] = "M/S".lower()
            if i == "Degrees Fahrenheit":
                df.loc[con, "obs"] = (df.loc[con, "obs"] + 459.67) * 5.0 / 9.0
                df.loc[con, "units"] = "K"
            if i == "Percent relative humidity":
                df.loc[con, "units"] = "%"
        return df
