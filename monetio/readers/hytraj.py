"""HYTRAJ Reader"""

import re
import numpy as np
import pandas as pd
from .base import PointReader, register_reader
from .drivers import FileUtility

@register_reader("hytraj")
class HYTRAJReader(PointReader):
    def open_dataset(self,
                     files,
                     taglist=None,
                     renumber=False,
                     verbose=False,
                     **kwargs):
        """
        Reads HYTRAJ tdump files.
        """
        file_list = FileUtility.expand_paths(files)
        return combine_dataset(file_list, taglist=taglist, renumber=renumber, verbose=verbose)

# -----------------------------------------------------------------------------
# Helper functions ported from monetio/models/hytraj.py
# -----------------------------------------------------------------------------

def combine_dataset(flist, taglist=None, renumber=False, verbose=False):
    usepid = False
    if isinstance(taglist, (tuple, list, np.ndarray)):
        if len(taglist) == len(flist):
            usepid = True
        else:
            if verbose:
                print("WARNING, taglist different length than flist. cannot use")
            taglist = None

    if not renumber:
        if not isinstance(taglist, (tuple, list, np.ndarray)):
            taglist = np.arange(1, len(flist) + 2, 1)
            usepid = True

    maxtrajnum = 0
    rval = None

    for iii, fname in enumerate(flist):
        traj = open_dataset_hytraj(fname)
        if usepid:
            traj["pid"] = taglist[iii]
        if renumber:
            traj["traj_num"] += maxtrajnum
        if iii == 0:
            rval = traj
        else:
            rval = pd.concat([rval, traj])
        maxtrajnum = np.max(rval.traj_num.unique())
    return rval

def open_dataset_hytraj(filename):
    # Use FileUtility to get filesystem and open
    fs = FileUtility.get_fs(filename)
    # pd.read_csv can usually take a file-like object if opened in text mode?
    # Original used open(filename). Default is text 'r'.
    # fsspec open returns bytes by default unless mode='r' which might be bytes or text depending on implementation.
    # Safe to use 'r' for text if supported, or 'rb' and decode.
    # TextIOWrapper is safer.

    # fsspec open(..., "r") often returns text mode.
    tdump = fs.open(filename, "r")

    # However, get_metinfo uses seek(0) and readline().
    traj = get_traj(tdump)
    tdump.close()
    return traj

def get_metinfo(tdump):
    tdump.seek(0)
    dim1 = tdump.readline().strip().replace(" ", "")
    dim1 = np.array(list(dim1))
    metinfo = []
    a = 0
    while a < int(dim1[0]):
        tmp = re.sub(r"\s+", ",", tdump.readline().strip())
        metinfo.append(tmp)
        a += 1
    return metinfo

def get_startlocs(tdump):
    tdump.seek(0)
    _ = get_metinfo(tdump)
    dim2 = list(tdump.readline().strip().split(" "))
    start_locs = []
    b = 0
    while b < int(dim2[0]):
        tmp2 = re.sub(r"\s+", ",", tdump.readline().strip())
        tmp2 = tmp2.split(",")
        start_locs.append(tmp2)
        b += 1
    heads = ["year", "month", "day", "hour", "latitude", "longitude", "altitude"]
    stlocs = pd.DataFrame(np.array(start_locs), columns=heads)
    cols = ["year", "month", "day", "hour"]
    stlocs["time"] = stlocs[cols].apply(lambda row: " ".join(row.values.astype(str)), axis=1)
    stlocs = stlocs.drop(cols, axis=1)
    stlocs = stlocs[["time", "latitude", "longitude", "altitude"]]
    stlocs["time"] = stlocs.apply(lambda row: time_str_fixer(row["time"]), axis=1)
    stlocs["time"] = pd.to_datetime(stlocs["time"], format="%y %m %d %H")
    return stlocs

def time_str_fixer(timestr):
    if isinstance(timestr, str):
        temp = timestr.split()
        year = str(int(temp[0])).zfill(2)
        month = str(int(temp[1])).zfill(2)
        temp[0] = year
        temp[1] = month
        rval = str.join(" ", temp)
    else:
        rval = timestr
    return rval

def get_traj(tdump):
    tdump.seek(0)
    _ = get_startlocs(tdump)
    varibs = re.sub(r"\s+", ",", tdump.readline().strip())
    varibs = varibs.split(",")
    variables = varibs[1:]
    heads = (
        [
            "traj_num",
            "met_grid",
            "forecast_hour",
            "traj_age",
            "latitude",
            "longitude",
            "altitude",
        ]
        + variables
        + ["time"]
    )

    def dateparse(row):
        slist = [row[2], row[3], row[4], row[5], row[6]]
        tstr = " ".join(slist)
        tstr = time_str_fixer(tstr)
        tdate = pd.to_datetime(tstr, format="%y %m %d %H %M")
        return tdate

    dhash = {
        0: int,
        1: int,
        2: str,
        3: str,
        4: str,
        5: str,
        6: str,
        7: float,
        8: float,
        9: float,
        10: float,
        11: float,
    }
    # pd.read_csv accepts file-like object
    traj = pd.read_csv(tdump, header=None, sep=r"\s+", dtype=dhash)
    traj["time"] = traj.apply(lambda row: dateparse(row), axis=1)
    traj = traj.drop([2, 3, 4, 5, 6], axis=1)
    traj.columns = heads
    neworder = [
        "time",
        "traj_num",
        "met_grid",
        "forecast_hour",
        "traj_age",
        "latitude",
        "longitude",
        "altitude",
    ] + variables
    traj = traj[neworder]
    traj.columns = map(str.lower, traj.columns)
    return traj
