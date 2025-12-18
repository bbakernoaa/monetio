"""CEMS Reader"""

import datetime
import os

import pandas as pd

from .base import PointReader, register_reader


@register_reader("cems")
class CEMSReader(PointReader):
    def open_dataset(
        self,
        rdate,
        states=["md"],
        download=False,
        verbose=True,
        files=None,  # Support local files directly
        **kwargs,
    ):
        """
        Reads CEMS data.
        """
        c = CEMS()

        if files:
            # If explicit files are provided
            if isinstance(files, str):
                files = [files]

            dfs = []
            for f in files:
                df = c.load(f, verbose=verbose)
                dfs.append(df)

            if not dfs:
                return pd.DataFrame()
            return pd.concat(dfs)

        else:
            return c.add_data(rdate, states=states, download=download, verbose=verbose)


# -----------------------------------------------------------------------------
# Helper functions ported from monetio/obs/cems_mod.py
# -----------------------------------------------------------------------------


def getdegrees(degrees, minutes, seconds):
    return degrees + minutes / 60.0 + seconds / 3600.00


def addmonth(dt):
    month = dt.month + 1
    year = dt.year
    day = dt.day
    hour = dt.hour
    if month > 12:
        year = dt.year + 1
        month = month - 12
        if day == 31 and month in [4, 6, 9, 11]:
            day = 30
        if month == 2 and day in [29, 30, 31]:
            if year % 4 == 0:
                day = 29
            else:
                day = 28
    return datetime.datetime(year, month, day, hour)


def get_date_fmt(date, verbose=False):
    temp = date.split("-")
    if len(temp[0]) == 4:
        fmt = "%Y-%m-%d %H"
    else:
        fmt = "%m-%d-%Y %H"
    return fmt


class CEMS:
    def __init__(self):
        self.efile = None
        self.url = "ftp://newftp.epa.gov/DmDnLoad/emissions/"
        self.lb2kg = 0.453592
        self.info = "Data from continuous emission monitoring systems (CEMS)\n"
        self.df = pd.DataFrame()
        self.namehash = {}

    def add_data(self, rdate, states=["md"], download=False, verbose=True):
        if isinstance(states, str):
            states = [states]
        if isinstance(rdate, list):
            r1 = rdate[0]
            r2 = rdate[1]
            rdatelist = [r1]
            done = False
            iii = 0
            while not done:
                r3 = addmonth(rdatelist[-1])
                if r3 <= r2:
                    rdatelist.append(r3)
                else:
                    done = True
                if iii > 100:
                    done = True
                iii += 1
        else:
            rdatelist = [rdate]
        for rd in rdatelist:
            if verbose:
                print("getting data", rd)
            for st in states:
                url = self.retrieve(rd, st, download=download, verbose=verbose)
                self.load(url, verbose=verbose)
        return self.df

    def retrieve(self, rdate, state, download=True, verbose=False):
        efile = "empty"
        ftpsite = self.url
        ftpsite += "hourly/"
        ftpsite += "monthly/"
        ftpsite += rdate.strftime("%Y") + "/"
        fname = rdate.strftime("%Y") + state + rdate.strftime("%m") + ".zip"
        if not download:
            efile = ftpsite + fname
        if not os.path.isfile(fname):
            # CEMS requires manual download usually due to FTP issues or explicit requests
            # Original code warns about download not supported
            efile = ftpsite + fname
            print("WARNING: Downloading file not supported at this time")
            print(efile)
        else:
            print("file exists " + fname)
            efile = fname
        self.info += "File retrieved :" + efile + "\n"
        return efile

    def columns_rename(self, columns, verbose=False):
        rcolumn = []
        for ccc in columns:
            if "facility" in ccc.lower() and "name" in ccc.lower():
                rcolumn = self.rename(ccc, "facility_name", rcolumn, verbose)
            elif "orispl" in ccc.lower():
                rcolumn = self.rename(ccc, "orispl_code", rcolumn, verbose)
            elif "facility" in ccc.lower() and "id" in ccc.lower():
                rcolumn = self.rename(ccc, "fac_id", rcolumn, verbose)
            elif "so2" in ccc.lower() and "lbs" in ccc.lower() and "rate" not in ccc.lower():
                rcolumn = self.rename(ccc, "so2_lbs", rcolumn, verbose)
            elif "nox" in ccc.lower() and "lbs" in ccc.lower() and "rate" not in ccc.lower():
                rcolumn = self.rename(ccc, "nox_lbs", rcolumn, verbose)
            elif "co2" in ccc.lower() and "short" in ccc.lower() and "tons" in ccc.lower():
                rcolumn = self.rename(ccc, "co2_short_tons", rcolumn, verbose)
            elif "date" in ccc.lower():
                rcolumn = self.rename(ccc, "date", rcolumn, verbose)
            elif "hour" in ccc.lower():
                rcolumn = self.rename(ccc, "hour", rcolumn, verbose)
            elif "lat" in ccc.lower():
                rcolumn = self.rename(ccc, "latitude", rcolumn, verbose)
            elif "lon" in ccc.lower():
                rcolumn = self.rename(ccc, "longitude", rcolumn, verbose)
            elif "state" in ccc.lower():
                rcolumn = self.rename(ccc, "state_name", rcolumn, verbose)
            else:
                rcolumn.append(ccc.strip().lower())
        return rcolumn

    def rename(self, ccc, newname, rcolumn, verbose):
        self.namehash[newname] = ccc
        rcolumn.append(newname)
        if verbose:
            print(ccc + " to " + newname)
        return rcolumn

    def add_info(self, dftemp):
        # Placeholder for metadata merging (cemsinfo.csv)
        # Original code read a local CSV in `monetio/data`
        # We assume it might fail if file missing
        return dftemp

    def load(self, efile, verbose=True):
        dftemp = pd.read_csv(efile, sep=",", index_col=False, header=0)
        columns = list(dftemp.columns.values)
        columns = self.columns_rename(columns, verbose)
        dftemp.columns = columns

        dfmt = get_date_fmt(dftemp["date"][0], verbose=verbose)
        dftime = dftemp.apply(
            lambda x: datetime.datetime.strptime("{} {}".format(x["date"], x["hour"]), dfmt), axis=1
        )
        dftemp = pd.concat([dftime, dftemp], axis=1)
        dftemp.rename(columns={0: "time local"}, inplace=True)
        dftemp.drop(["date", "hour"], axis=1, inplace=True)

        dftemp = self.add_info(dftemp)

        if "year" in columns:
            dftemp.drop(["year"], axis=1, inplace=True)

        if self.df.empty:
            self.df = dftemp
        else:
            self.df = pd.concat([self.df, dftemp])  # Fixed append

        return dftemp
