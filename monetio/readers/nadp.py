"""NADP Reader"""

import pandas as pd
from numpy import nan

from .base import PointReader, register_reader


@register_reader("nadp")
class NADPReader(PointReader):
    def open_dataset(
        self, dates, network="NTN", siteid=None, weekly=True, as_xarray=False, **kwargs
    ):
        """
        Reads NADP data.
        """
        n = NADP()
        df = n.add_data(dates, network=network, siteid=siteid, weekly=weekly)

        df = self.harmonize(df)
        if as_xarray:
            return self.to_xarray(df)

        return df


# -----------------------------------------------------------------------------
# Helper functions ported from monetio/obs/nadp.py
# -----------------------------------------------------------------------------


class NADP:
    def __init__(self):
        self.weekly = True
        self.network = None
        self.df = pd.DataFrame()
        self.objtype = "NADP"
        self.url = None

    def build_url(self, network="NTN", siteid=None):
        baseurl = "http://nadp.slh.wisc.edu/datalib/"
        siteid = (siteid.upper() + "-") if siteid is not None else ""
        if network.lower() == "amnet":
            url = "http://nadp.slh.wisc.edu/datalib/AMNet/AMNet-All.zip"
        elif network.lower() == "amon":
            url = "http://nadp.slh.wisc.edu/dataLib/AMoN/csv/all-ave.csv"
        elif network.lower() == "airmon":
            url = "http://nadp.slh.wisc.edu/datalib/AIRMoN/AIRMoN-ALL.csv"
        else:
            if self.weekly:
                url = (
                    baseurl + network.lower() + "/weekly/" + siteid + network.upper() + "-All-w.csv"
                )
            else:
                url = (
                    baseurl + network.lower() + "/annual/" + siteid + network.upper() + "-All-a.csv"
                )
        return url

    def read_ntn(self, url):
        df = pd.read_csv(url, parse_dates=[2, 3])
        df.columns = [i.lower() for i in df.columns]
        df.rename(columns={"dateon": "time", "dateoff": "time_off"}, inplace=True)
        # Load meta
        try:
            meta = pd.read_csv("https://bit.ly/2sPMvaO")
        except Exception:
            # Fallback path logic omitted as we assume web access or mock
            meta = pd.DataFrame(columns=["siteid", "latitude", "longitude"])

        meta.columns = [i.lower() for i in meta.columns]
        if "startdate" in meta.columns:
            meta.drop(["startdate", "stopdate"], axis=1, inplace=True)

        dfn = pd.merge(df, meta, on="siteid", how="left")
        dfn.dropna(subset=["latitude", "longitude"], inplace=True)

        for col in ["mg", "br", "so4", "cl", "no3", "nh4", "k", "na", "ca"]:
            flag = "flag" + col
            if flag in dfn.columns:
                dfn.loc[(dfn[flag] == "<") | (dfn[col] < 0), col] = nan
        return dfn

    def read_mdn(self, url):
        df = pd.read_csv(url, parse_dates=[1, 2])
        df.columns = [i.lower() for i in df.columns]
        df.rename(columns={"dateon": "time", "dateoff": "time_off"}, inplace=True)
        try:
            meta = pd.read_csv("https://bit.ly/2Lq6kgq")
            meta.drop(["startdate", "stopdate"], axis=1, inplace=True)
        except Exception:
            meta = pd.DataFrame(columns=["siteid", "latitude", "longitude"])

        meta.columns = [i.lower() for i in meta.columns]
        dfn = pd.merge(df, meta, on="siteid", how="left")
        dfn.dropna(subset=["latitude", "longitude"], inplace=True)
        dfn.loc[dfn.qr == "C", ["rgppt", "svol", "subppt", "hgconc", "hgdep"]] = nan
        return dfn

    def read_airmon(self, url):
        df = pd.read_csv(url, parse_dates=[2, 3])
        df.columns = [i.lower() for i in df.columns]
        df.rename(columns={"dateon": "time", "dateoff": "time_off"}, inplace=True)
        try:
            meta = pd.read_csv("https://bit.ly/2xMlgTW")
            meta.drop(["startdate", "stopdate"], axis=1, inplace=True)
        except Exception:
            meta = pd.DataFrame(columns=["siteid", "latitude", "longitude"])

        meta.columns = [i.lower() for i in meta.columns]
        dfn = pd.merge(df, meta, on="siteid", how="left")
        dfn.dropna(subset=["latitude", "longitude"], inplace=True)
        if "qrcode" in dfn.columns:
            cols = [
                "subppt",
                "pptnws",
                "pptbel",
                "svol",
                "ca",
                "mg",
                "k",
                "na",
                "nh4",
                "no3",
                "cl",
                "so4",
                "po4",
                "phlab",
                "phfield",
                "conduclab",
                "conducfield",
            ]
            dfn.loc[dfn.qrcode == "C", cols] = nan
        return dfn

    def read_amon(self, url):
        df = pd.read_csv(url, parse_dates=[2, 3])
        df.columns = [i.lower() for i in df.columns]
        df.rename(columns={"startdate": "time", "enddate": "time_off"}, inplace=True)
        try:
            meta = pd.read_csv("https://bit.ly/2sJmkCg")
            meta.drop(["startdate", "stopdate"], axis=1, inplace=True)
        except Exception:
            meta = pd.DataFrame(columns=["siteid", "latitude", "longitude"])

        meta.columns = [i.lower() for i in meta.columns]
        dfn = pd.merge(df, meta, on="siteid", how="left")
        dfn.dropna(subset=["latitude", "longitude"], inplace=True)
        if "qr" in dfn.columns:
            dfn.loc[dfn.qr == "C", ["airvol", "conc"]] = nan
        return dfn

    def read_amnet(self, url):
        df = pd.read_csv(url, parse_dates=[2, 3])
        df.columns = [i.lower() for i in df.columns]
        df.rename(columns={"startdate": "time", "enddate": "time_off"}, inplace=True)
        try:
            meta = pd.read_csv("https://bit.ly/2sJmkCg")
            meta.drop(["startdate", "stopdate"], axis=1, inplace=True)
        except Exception:
            meta = pd.DataFrame(columns=["siteid", "latitude", "longitude"])

        meta.columns = [i.lower() for i in meta.columns]
        dfn = pd.merge(df, meta, on="siteid", how="left")
        dfn.dropna(subset=["latitude", "longitude"], inplace=True)
        if "qr" in dfn.columns:
            dfn.loc[dfn.qr == "C", ["airvol", "conc"]] = nan
        return dfn

    def add_data(self, dates, network="NTN", siteid=None, weekly=True):
        url = self.build_url(network=network, siteid=siteid)
        n = network.lower()
        if n == "ntn":
            df = self.read_ntn(url)
        elif n == "mdn":
            df = self.read_mdn(url)
        elif n == "amon":
            df = self.read_amon(url)
        elif n == "airmon":
            df = self.read_airmon(url)
        else:
            df = self.read_amnet(url)

        self.df = df
        if "time" in self.df.columns and "time_off" in self.df.columns:
            self.df = self.df.loc[(self.df.time >= dates.min()) & (self.df.time_off <= dates.max())]
        return self.df
