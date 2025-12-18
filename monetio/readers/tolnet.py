"""TOLNet Reader"""

import os

import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader
from .drivers import FileUtility


@register_reader("tolnet")
class TOLNetReader(GriddedReader):
    def open_dataset(self, files, **kwargs):
        """
        Reads TOLNet HDF5 files.
        """
        # Expand paths
        file_list = FileUtility.expand_paths(files)

        dsets = []
        t = TOLNet()
        for i in file_list:
            dsets.append(t.add_data(i))

        if not dsets:
            return xr.Dataset()

        if len(dsets) == 1:
            return dsets[0]
        else:
            return xr.concat(dsets, dim="time")


# -----------------------------------------------------------------------------
# Helper functions ported from monetio/profile/tolnet.py
# -----------------------------------------------------------------------------


class TOLNet:
    def __init__(self):
        self.objtype = "TOLNET"
        self.cwd = os.getcwd()
        self.dates = pd.date_range(start="2017-09-25", end="2017-09-26", freq="H")
        self.dset = None
        self.daily = False

    def add_data(self, fname):
        from h5py import File

        # FileUtility logic for HDF5?
        # h5py can take a file-like object (bytes)

        fs = FileUtility.get_fs(fname)
        # h5py requires seekable file
        f_obj = fs.open(fname, "rb")

        f = File(f_obj, "r")
        atts = f["INSTRUMENT_ATTRIBUTES"]
        data = f["DATA"]
        self.dset = self.make_xarray_dataset(data, atts)
        f.close()
        # f_obj auto closed? Maybe not.
        f_obj.close()

        return self.dset

    @staticmethod
    def make_xarray_dataset(data, atts):
        from numpy import array, ndarray

        alt = data["ALT"][:].squeeze()
        altvars = ["AirND", "AirNDUncert", "ChRange", "Press", "Temp", "TempUncert", "PressUncert"]
        tseries = pd.Series(data["TIME_MID_UT_UNIX"][:].squeeze())
        time = pd.Series(pd.to_datetime(tseries, unit="ms"), name="time")
        ovars = ["O3MR", "O3ND", "O3NDUncert", "O3MRUncert", "O3NDResol", "Precision"]

        dataset = xr.Dataset()
        dataset["z"] = (("z"), alt)
        dataset["time"] = (("time"), time)
        dataset["x"] = (("x"), [0])
        dataset["y"] = (("y"), [0])

        for i in ovars:
            if i in data:
                if data[i].shape == (len(alt), len(time)):
                    dataset[i] = (("z", "time"), data[i][:])
                elif data[i].shape == (len(alt), 1):
                    dataset[i] = (("z"), data[i][:].squeeze())
                else:
                    dataset[i] = (("time"), data[i][:].squeeze())
                dataset[i] = dataset[i].where(dataset[i] > -990)

        for i in altvars:
            if i in data:
                dataset[i] = (("z"), data[i][:].squeeze())

        for i in list(atts.attrs.keys()):
            if isinstance(atts.attrs[i], list) or isinstance(atts.attrs[i], ndarray):
                dataset.attrs[i] = atts.attrs[i][0]
            else:
                dataset.attrs[i] = atts.attrs[i]

        try:
            a, b = dataset.Location_Latitude.decode("ascii").split()
            if b == "S":
                latitude = -1 * float(a)
            else:
                latitude = float(a)

            a, b = dataset.Location_Longitude.decode("ascii").split()
            if b == "W":
                longitude = -1 * float(a)
            else:
                longitude = float(a)

            dataset.coords["latitude"] = (("y", "x"), array(latitude).reshape(1, 1))
            dataset.coords["longitude"] = (("y", "x"), array(longitude).reshape(1, 1))
        except Exception:
            pass

        return dataset
