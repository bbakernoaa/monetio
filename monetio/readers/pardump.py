"""PARDUMP Reader"""

import datetime
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

from .base import PointReader, register_reader
from .drivers import FileUtility


@register_reader("pardump")
class PardumpReader(PointReader):
    fixed_location = False

    def open_dataset(
        self,
        files: Optional[Union[str, List[str]]] = None,
        dates: Optional[Any] = None,
        drange=None,
        century=2000,
        verbose=False,
        **kwargs,
    ):
        """
        Reads HYSPLIT PARDUMP binary files.
        """
        res = self._prepare_files(files, dates, **kwargs)
        if isinstance(res, (pd.DataFrame, xr.Dataset)):
            return res

        file_list = FileUtility.expand_paths(res)

        dfs = []
        for f in file_list:
            pdump = Pardump(f)
            df = pdump.read(drange=drange, century=century, verbose=verbose)
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        if len(dfs) == 1:
            df = dfs[0]
        else:
            df = pd.concat(dfs)

        return df


# -----------------------------------------------------------------------------
# Helper functions ported from monetio/models/pardump.py
# -----------------------------------------------------------------------------


class Pardump:
    def __init__(self, fname="PARINIT"):
        self.fname = fname
        tp1 = ">f4"
        tp2 = ">i4"
        tp3 = ">i8"

        self.hdr_dt = np.dtype(
            [
                ("padding", tp2),
                ("parnum", tp2),
                ("pollnum", tp2),
                ("year", tp2),
                ("month", tp2),
                ("day", tp2),
                ("hour", tp2),
                ("minute", tp2),
            ]
        )

        self.pardt = np.dtype(
            [
                ("p1", tp2),
                ("p2", tp2),
                ("pmass", tp1),
                ("p3", tp3),
                ("lat", tp1),
                ("lon", tp1),
                ("ht", tp1),
                ("su", tp1),
                ("sv", tp1),
                ("sx", tp1),
                ("p4", tp3),
                ("age", tp2),
                ("dist", tp2),
                ("poll", tp2),
                ("mgrid", tp2),
                ("sorti", tp2),
            ]
        )

    def read(self, drange=None, verbose=False, century=2000, sorti=None):
        imax = 100000
        parframe_all = pd.DataFrame()

        fs = FileUtility.get_fs(self.fname)
        with fs.open(self.fname, "rb") as fpoint:
            iii = 0
            testf = True
            while testf:
                hdata = np.fromfile(fpoint, dtype=self.hdr_dt, count=1)
                if verbose:
                    print("Record Header ", hdata)
                if not hdata.size:
                    if verbose:
                        print("Done reading ", self.fname)
                    break
                if hdata["year"] < 1000:
                    year = hdata["year"] + century
                else:
                    year = hdata["year"]
                pdate = datetime.datetime(
                    int(year),
                    int(hdata["month"]),
                    int(hdata["day"]),
                    int(hdata["hour"]),
                    int(hdata["minute"]),
                )
                parnum = hdata["parnum"]
                data = np.fromfile(fpoint, dtype=self.pardt, count=parnum[0])
                np.fromfile(fpoint, dtype=">i4", count=1)
                if verbose:
                    print("Date ", pdate, " **** ", drange)

                testdate = False
                if not drange:
                    testdate = True
                elif pdate >= drange[0] and pdate <= drange[1]:
                    testdate = True

                if testdate:
                    ndata = data.byteswap().newbyteorder()
                    par_frame = pd.DataFrame.from_records(ndata)
                    par_frame.drop(["p1", "p2", "p3", "p4"], inplace=True, axis=1)
                    par_frame.drop(["su", "sv", "sx", "mgrid"], inplace=True, axis=1)
                    par_frame = par_frame.loc[par_frame["lat"] != 0]

                    if sorti:
                        par_frame = par_frame.loc[par_frame["sorti"].isin(sorti)]
                    par_frame["date"] = pdate
                    if iii == 0:
                        parframe_all = par_frame.copy()
                    else:
                        parframe_all = pd.concat([parframe_all, par_frame], axis=0)

                iii += 1

                if drange:
                    if pdate > drange[1]:
                        testf = False
                        if verbose:
                            print("Past date. Closing file.", drange[1], pdate)
                if iii > imax:
                    print("Read pardump. Limited to" + str(imax) + "  iterations. Stopping")
                    testf = False

        if not parframe_all.empty:
            parframe_all = pd.concat([parframe_all], keys=[self.fname])

        return parframe_all
