"""CRN Reader"""

import os

import dask
import dask.dataframe as dd
import pandas as pd

from .base import PointReader, register_reader
from .drivers import FileUtility


@register_reader("crn")
class CRNReader(PointReader):
    def open_dataset(
        self,
        dates,
        daily=False,
        sub_hourly=False,
        download=False,
        latlonbox=None,
        as_xarray=True,
        **kwargs,
    ):
        """
        Reads CRN data.
        """
        c = CRN()
        df = c.add_data(
            dates,
            daily=daily,
            sub_hourly=sub_hourly,
            download=download,
            latlonbox=latlonbox,
        )

        df = self.harmonize(df)
        if as_xarray:
            return self.to_xarray(df)

        return df


# -----------------------------------------------------------------------------
# Helper functions ported from monetio/obs/crn.py
# -----------------------------------------------------------------------------


class CRN:
    def __init__(self):
        self.dates = None
        self.daily = False
        self.df = pd.DataFrame()
        self.baseurl = "https://www1.ncdc.noaa.gov/pub/data/uscrn/products/"
        self.monitor_df = None
        # Columns definitions omitted for brevity
        self.hcols = [
            "WBANNO",
            "UTC_DATE",
            "UTC_TIME",
            "LST_DATE",
            "LST_TIME",
            "CRX_VN",
            "LONGITUDE",
            "LATITUDE",
            "T_CALC",
            "T_AVG",
            "T_MAX",
            "T_MIN",
            "P_CALC",
            "SOLARAD",
            "SOLARAD_FLAG",
            "SOLARAD_MAX",
            "SOLARAD_MAX_FLAG",
            "SOLARAD_MIN",
            "SOLARAD_MIN_FLAG",
            "SUR_TEMP_TYPE",
            "SUR_TEMP",
            "SUR_TEMP_FLAG",
            "SUR_TEMP_MAX",
            "SUR_TEMP_MAX_FLAG",
            "SUR_TEMP_MIN",
            "SUR_TEMP_MIN_FLAG",
            "RH_AVG",
            "RH_AVG_FLAG",
            "SOIL_MOISTURE_5",
            "SOIL_MOISTURE_10",
            "SOIL_MOISTURE_20",
            "SOIL_MOISTURE_50",
            "SOIL_MOISTURE_100",
            "SOIL_TEMP_5",
            "SOIL_TEMP_10",
            "SOIL_TEMP_20",
            "SOIL_TEMP_50",
            "SOIL_TEMP_100",
        ]
        self.dcols = [
            "WBANNO",
            "LST_DATE",
            "CRX_VN",
            "LONGITUDE",
            "LATITUDE",
            "T_MAX",
            "T_MIN",
            "T_MEAN",
            "T_AVG",
            "P_CALC",
            "SOLARAD",
            "SUR_TEMP_TYPE",
            "SUR_TEMP_MAX",
            "SUR_TEMP_MAX",
            "SUR_TEMP_MIN",
            "SUR_TEMP_AVG",
            "RH_MAX",
            "RH_MIN",
            "RH_AVG",
            "SOIL_MOISTURE_5",
            "SOIL_MOISTURE_10",
            "SOIL_MOISTURE_20",
            "SOIL_MOISTURE_50",
            "SOIL_MOISTURE_100",
            "SOIL_TEMP_5",
            "SOIL_TEMP_10",
            "SOIL_TEMP_20",
            "SOIL_TEMP_50",
            "SOIL_TEMP_100",
        ]
        self.shcols = [
            "WBANNO",
            "UTC_DATE",
            "UTC_TIME",
            "LST_DATE",
            "LST_TIME",
            "CRX_VN",
            "LONGITUDE",
            "LATITUDE",
            "T_MEAN",
            "P_CALC",
            "SOLARAD",
            "SOLARAD_FLAG",
            "SUR_TEMP_AVG",
            "SUR_TEMP_TYPE",
            "SUR_TEMP_FLAG",
            "RH_AVG",
            "RH_FLAG",
            "SOIL_MOISTURE_5",
            "SOIL_TEMP_5",
            "WETNESS",
            "WET_FLAG",
            "WIND",
            "WIND_FLAG",
        ]

    def load_file(self, url):
        nanvals = [-99999, -9999.0]
        # Check url type via string (or regex if robust needed)
        # The filename pattern indicates type
        if "CRND0103" in url:
            cols = self.dcols
            parse_dates = {"time_local": [1]}
            self.daily = True
        elif "CRNS0101" in url:
            cols = self.shcols
            parse_dates = {
                "time": ["UTC_DATE", "UTC_TIME"],
                "time_local": ["LST_DATE", "LST_TIME"],
            }
        else:
            cols = self.hcols
            parse_dates = {
                "time": ["UTC_DATE", "UTC_TIME"],
                "time_local": ["LST_DATE", "LST_TIME"],
            }

        fs = FileUtility.get_fs(url)
        with fs.open(url, "r") as f:
            df = pd.read_csv(
                f,
                sep=r"\s+",
                names=cols,
                parse_dates=parse_dates,
                na_values=nanvals,
            )
        return df

    def build_url(self, year, state, site, vector, daily=False, sub_hourly=False):
        if daily:
            beginning = self.baseurl + "daily01/" + year + "/"
            fname = "CRND0103-"
        elif sub_hourly:
            beginning = self.baseurl + "subhourly01/" + year + "/"
            fname = "CRNS0101-05-"
        else:
            beginning = self.baseurl + "hourly02/" + year + "/"
            fname = "CRNH0203-"
        rest = year + "-" + state + "_" + site + "_" + vector + ".txt"
        url = beginning + fname + rest
        fname = fname + rest
        return url, fname

    def check_url(self, url):
        fs = FileUtility.get_fs(url)
        try:
            # For http, exists calls HEAD. For s3/local, it checks existence.
            return fs.exists(url)
        except Exception:
            return False

    def build_urls(self, monitors, dates, daily=False, sub_hourly=False):
        years = pd.DatetimeIndex(dates).year.unique().astype(str)
        urls = []
        fnames = []
        for i in monitors.index:
            for y in years:
                state = monitors.iloc[i].STATE
                site = monitors.iloc[i].LOCATION.replace(" ", "_")
                vector = monitors.iloc[i].VECTOR.replace(" ", "_")
                url, fname = self.build_url(
                    y, state, site, vector, daily=daily, sub_hourly=sub_hourly
                )
                if self.check_url(url):
                    urls.append(url)
                    fnames.append(fname)
        return urls, fnames

    def retrieve(self, url, fname):
        fs = FileUtility.get_fs(url)
        if not os.path.isfile(fname):
            print("Retrieving: " + fname)
            print(url)
            fs.get(url, fname)
        else:
            print("File Exists: " + fname)

    def get_monitor_df(self):
        try:
            import monetio

            path = os.path.join(os.path.dirname(monetio.__file__), "data", "stations.tsv")
            self.monitor_df = pd.read_csv(path, delimiter="\t")
        except Exception:
            print("Could not load stations.tsv")
            self.monitor_df = pd.DataFrame(
                columns=[
                    "STATE",
                    "LOCATION",
                    "VECTOR",
                    "WBANNO",
                    "LATITUDE",
                    "LONGITUDE",
                ]
            )

    def add_data(self, dates, daily=False, sub_hourly=False, download=False, latlonbox=None):
        if self.monitor_df is None:
            self.get_monitor_df()

        if latlonbox is not None:
            mdf = self.monitor_df
            con = (
                (mdf.LATITUDE >= latlonbox[0])
                & (mdf.LATITUDE <= latlonbox[2])
                & (mdf.LONGITUDE >= latlonbox[1])
                & (mdf.LONGITUDE <= latlonbox[3])
            )
            monitors = mdf.loc[con].copy()
        else:
            monitors = self.monitor_df.copy()

        urls, fnames = self.build_urls(monitors, dates, daily=daily, sub_hourly=sub_hourly)

        if download:
            for url, fname in zip(urls, fnames):
                self.retrieve(url, fname)
            # After download, files are local
            # Original code used delayed(load_file)(fname)
            # Here we just pass fnames (which are local paths)
            dfs = [dask.delayed(self.load_file)(i) for i in fnames]
        else:
            dfs = [dask.delayed(self.load_file)(i) for i in urls]

        dff = dd.from_delayed(dfs)
        self.df = dff.compute()

        self.df = pd.merge(self.df, monitors, how="left", on=["WBANNO", "LATITUDE", "LONGITUDE"])

        if not self.df.columns.isin(["time"]).max():
            if "time_local" in self.df.columns and "GMT_OFFSET" in self.df.columns:
                self.df["time"] = self.df.time_local + pd.to_timedelta(self.df.GMT_OFFSET, unit="h")

        self.df.rename(columns={"WBANNO": "siteid"}, inplace=True)
        self.df.columns = [i.lower() for i in self.df.columns]

        return self.df
