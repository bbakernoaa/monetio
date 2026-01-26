"""ICARTT Reader"""

import datetime
import pandas as pd
import xarray as xr
from numpy import nan
from .base import GriddedReader, register_reader
from .drivers import FileUtility


@register_reader("icartt")
class ICARTTReader(GriddedReader):
    def open_dataset(self, files, **kwargs):
        """
        Reads ICARTT files.
        """
        file_list = FileUtility.expand_paths(files)

        ds_list = []
        for f in file_list:
            o = Dataset(f)
            ds = class_to_xarray(o)
            ds_list.append(ds)

        if not ds_list:
            return xr.Dataset()

        if len(ds_list) == 1:
            return ds_list[0]
        else:
            return xr.concat(ds_list, dim="time")


# -----------------------------------------------------------------------------
# Helper functions ported from monetio/profile/icartt.py
# -----------------------------------------------------------------------------


def var_to_da(o, var_name, time):
    unit = o.units(var_name)
    bad_val = nan
    vals = o[var_name]
    name = var_name
    if "Latitude" in var_name:
        name = "latitude"
        unit = "degrees_north"
    if "Longitude" in var_name:
        name = "longitude"
        unit = "degrees_east"
    da = xr.DataArray(vals, coords=[time], dims=["time"])
    da.name = name
    da.attrs["units"] = unit
    da.attrs["missing_value"] = bad_val
    return da


def class_to_xarray(o, time_str="Time_Start"):
    time_index = pd.to_datetime(o.times)
    das = {}
    for i in o.varnames:
        if i != "Time_Start":
            das[i] = var_to_da(o, i, time_index)
    ds = xr.Dataset(das)
    ds.attrs["source"] = o.dataSource
    ds.attrs["Date Revised"] = pd.to_datetime(o.dateRevised).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    ds.attrs["mission"] = o.mission
    ds.attrs["organization"] = o.organization
    ds.attrs["PI"] = o.PI
    if len(o.NCOM) > 1:
        for i in o.NCOM[:-1]:
            try:
                name = i.split(":")[0].strip()
                val = i.split(":")[1].strip()
                ds.attrs[name] = val
            except IndexError:
                pass
    return ds


class Variable:
    @property
    def desc(self):
        return self.splitChar.join([self.name, self.units, self.units])

    def __init__(self, name, units, scale=1.0, miss=-9999999):
        self.name = name
        self.units = units
        self.scale = scale
        self.miss = str(miss)
        self.splitChar = ","


class Dataset:
    @property
    def nheader(self):
        total = 12 + self.ndvar + 1 + self.nscom + 1 + self.nncom
        if self.format == 2110:
            total += self.nauxvar + 5
        return total

    @property
    def ndvar(self):
        return len(self.DVAR)

    @property
    def nauxvar(self):
        return len(self.AUXVAR)

    @property
    def nvar(self):
        return self.ndvar + 1

    @property
    def nscom(self):
        return len(self.SCOM)

    @property
    def nncom(self):
        return len(self.NCOM)

    @property
    def VAR(self):
        return [self.IVAR] + self.DVAR

    @property
    def varnames(self):
        return [x.name for x in self.VAR]

    @property
    def times(self):
        return [
            self.dateValid + datetime.timedelta(seconds=x) for x in self[self.IVAR.name]
        ]

    def __getitem__(self, name):
        idx = self.index(name)
        if idx == -1:
            raise Exception(f"{name:s} not found in data")
        return [x[idx] for x in self.data]

    def units(self, name):
        res = [x.units for x in self.VAR if x.name == name]
        if len(res) == 0:
            res = [""]
        return res[0]

    def index(self, name):
        res = [i for i, x in enumerate(self.VAR) if x.name == name]
        if len(res) == 0:
            res = [-1]
        return res[0]

    def __readline(self, do_split=True):
        dmp = self.input_fhandle.readline().replace("\n", "").replace("\r", "")
        if do_split:
            dmp = [word.strip(" ") for word in dmp.split(self.splitChar)]
        return dmp

    def read_header(self):
        if self.input_fhandle.closed:
            self.input_fhandle = open(self.input_fhandle.name)

        self.format = int(self.__readline()[1])
        self.PI = self.__readline(do_split=False)
        self.organization = self.__readline(do_split=False)
        self.dataSource = self.__readline(do_split=False)
        self.mission = self.__readline(do_split=False)
        dmp = self.__readline()
        self.VOL = int(dmp[0])
        self.NVOL = int(dmp[1])
        dmp = self.__readline()
        self.dateValid = datetime.datetime.strptime(
            "".join([f"{x:s}" for x in dmp[0:3]]), "%Y%m%d"
        )
        self.dateRevised = datetime.datetime.strptime(
            "".join([f"{x:s}" for x in dmp[3:6]]), "%Y%m%d"
        )
        self.dataInterval = float(self.__readline()[0])
        dmp = self.__readline()
        self.IVAR = Variable(dmp[0], dmp[1])
        ndvar = int(self.__readline()[0])
        dvscale = [float(x) for x in self.__readline()]
        dvmiss = [x for x in self.__readline()]

        dmp = self.__readline()
        dvname = [dmp[0]]
        dvunits = [dmp[1]]

        for i in range(1, ndvar):
            dmp = self.__readline()
            dvname += [dmp[0]]
            dvunits += [dmp[1]]

        self.DVAR = [
            Variable(name, unit, scale, miss)
            for name, unit, scale, miss in zip(dvname, dvunits, dvscale, dvmiss)
        ]

        nscom = int(self.__readline()[0])
        self.SCOM = [self.__readline(do_split=False) for i in range(0, nscom)]
        nncom = int(self.__readline()[0])
        self.NCOM = [self.__readline(do_split=False) for i in range(0, nncom)]
        self.input_fhandle.close()

    def __nan_miss_float(self, raw):
        vals = []
        for i, x in enumerate(raw):
            v = x.replace(self.VAR[i].miss, "NaN")
            if "NaN" in v:
                v = "NaN"
            vals.append(float(v.strip()) * self.VAR[i].scale)
        return vals

    def read_data(self):
        if self.input_fhandle.closed:
            self.input_fhandle = open(self.input_fhandle.name)
        _ = [self.input_fhandle.readline() for _ in range(self.nheader)]
        self.data = [
            self.__nan_miss_float(line.split(self.splitChar))
            for line in self.input_fhandle
        ]
        self.input_fhandle.close()

    def read(self):
        self.read_header()
        self.read_data()

    def __init__(self, f=None, loadData=True):
        self.format = 1001
        self.revision = "0"
        self.dataID = "dataID"
        self.locationID = "locationID"
        self.PI = "Mustermann, Martin"
        self.organization = "Musterinstitut"
        self.dataSource = "Musterdatenprodukt"
        self.mission = "MUSTEREX"
        self.VOL = 1
        self.NVOL = 1
        self.dateValid = datetime.datetime.today()
        self.dateRevised = datetime.datetime.today()
        self.dataInterval = 0
        self.IVAR = Variable(
            "Time_Start", "seconds_from_0_hours_on_valid_date", 1.0, -9999999
        )
        self.DVAR = [
            Variable("Time_Stop", "seconds_from_0_hours_on_valid_date", 1.0, -9999999),
            Variable("Some_Variable", "ppbv", 1.0, -9999999),
        ]
        self.SCOM = []
        self.NCOM = []
        self.data = [[1.0, 2.0, 45.0], [2.0, 3.0, 36.0]]
        self.IBVAR = None
        self.AUXVAR = []
        self.splitChar = ","

        encoding = "utf-8"
        if f is not None:
            # Using FileUtility from driver logic? No, class takes filename.
            # We should adapt to use file handle or ensure f is path.
            # fsspec can handle paths.
            # But the logic uses `open(self.input_fhandle.name)`. This assumes local file!

            # Since I am in `monetio/readers/icartt.py`, I can import FileUtility.
            # And override the open calls.

            # BUT: self.input_fhandle is set to `open(f, ...)` initially if string.
            # We need to change that.

            self.filepath = f
            if isinstance(f, str):
                fs = FileUtility.get_fs(f)
                # We need to keep fs around or re-open
                # The read_header/read_data logic closes it.
                # And re-opens using `open(self.input_fhandle.name)`.
                # This logic is broken for S3 or remote files.

                # I will modify read_header and read_data to use self.filepath and FileUtility.

                self.input_fhandle = fs.open(f, "r", encoding=encoding)
            else:
                self.input_fhandle = f  # Assume it's a file-like object

            self.read_header()
            if loadData:
                self.read_data()
