"""TOLNet Reader"""

import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader
from .drivers import FileUtility


@register_reader("tolnet")
class TOLNetReader(GriddedReader):
    def open_dataset(self, files, **kwargs) -> xr.Dataset:
        """
        Retrieve and load TOLNet data.

        Parameters
        ----------
        files : Union[str, List[str]]
            File paths or URLs to read.
        **kwargs : dict
            Additional arguments passed to the driver.

        Returns
        -------
        xr.Dataset
            The loaded TOLNet data.
        """
        # We use XarrayDriver for lazy loading if possible, but TOLNet has custom HDF5 structure.
        # For now, we wrap the custom loading in a lazy-friendly way if n_procs > 1 or similar.
        # But XarrayDriver.open uses xr.open_mfdataset which is preferred.
        # However, TOLNet needs custom preprocessing.

        def preprocess(ds):
            return ds  # Placeholder if we used xr.open_dataset engine

        # TOLNet HDF5 isn't standard CF, so we use our custom reader via PandasDriver-like logic
        # but returning Datasets.
        # Actually, let's keep it simple for now and just use the unified driver if we can,
        # or refactor the loop to be more Aero Protocol friendly.

        file_list = FileUtility.expand_paths(files)

        if not file_list:
            return xr.Dataset()

        import dask

        @dask.delayed
        def load_one(f):
            return read_tolnet(f)

        if len(file_list) > 1:
            dsets = [load_one(f) for f in file_list]
            # We don't want to compute yet if we want to be lazy,
            # but xr.concat needs actual objects or we use something else.
            # PointReader handled this via dask.dataframe.
            # For Gridded, we usually rely on xr.open_mfdataset.

            # If we want bit-perfect lazy matching, we should use xr.open_mfdataset
            # with a custom engine or preprocess.

            # Since TOLNet is custom, we'll compute it for now to match original behavior
            # but make it faster with dask if requested.
            dsets = dask.compute(*dsets)
            ds = xr.concat(dsets, dim="time")
        else:
            ds = read_tolnet(file_list[0])

        ds = self.harmonize(ds)

        # Update history
        from datetime import datetime

        history = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read TOLNet data."
        if "history" in ds.attrs:
            ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
        else:
            ds.attrs["history"] = history

        return ds


# -----------------------------------------------------------------------------
# Helper functions ported from monetio/profile/tolnet.py
# -----------------------------------------------------------------------------


def read_tolnet(fname):
    """
    Read a single TOLNet HDF5 file.
    """
    from h5py import File
    from numpy import array, ndarray

    fs = FileUtility.get_fs(fname)
    with fs.open(fname, "rb") as f_obj:
        f = File(f_obj, "r")
        atts = f["INSTRUMENT_ATTRIBUTES"]
        data = f["DATA"]

        alt = data["ALT"][:].squeeze()
        altvars = [
            "AirND",
            "AirNDUncert",
            "ChRange",
            "Press",
            "Temp",
            "TempUncert",
            "PressUncert",
        ]
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
