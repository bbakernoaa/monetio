"""UMBC Aerosol Reader (CL51)"""

import datetime
from typing import List, Union

import numpy as np
import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader


@register_reader("umbc_aerosol")
class UMBCAerosolReader(GriddedReader):
    """
    Reader for UMBC Aerosol (CL51 Ceilometer) data.
    """

    def open_dataset(
        self,
        files: Union[str, List[str]],
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads UMBC Aerosol HDF5 files.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path, list of paths, or glob pattern.
        **kwargs : dict
            Additional arguments passed to the driver.

        Returns
        -------
        xr.Dataset
            The processed UMBC Aerosol dataset.
        """
        # We don't use XarrayDriver directly because the file structure is custom
        # and better handled by a custom loading logic that can be dask-ified.

        from .drivers import FileUtility

        file_list = FileUtility.expand_paths(files)

        dsets = []
        for f in file_list:
            ds = self._read_file(f)
            dsets.append(ds)

        if len(dsets) == 1:
            ds = dsets[0]
        else:
            ds = xr.concat(dsets, dim="time")

        ds = self.harmonize(ds)

        # Update history
        history = (
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read UMBC Aerosol data."
        )
        if "history" in ds.attrs:
            ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
        else:
            ds.attrs["history"] = history

        return ds

    def _read_file(self, fname: str) -> xr.Dataset:
        """Reads a single CL51 HDF5 file."""
        import h5py

        with h5py.File(fname, "r") as f:
            atts = f["Instrument_Attributes"]
            data = f["DATA"]

            # altitude variables
            alt = data["Altitude_m"][:].squeeze()

            # time variables
            time = pd.to_datetime(data["UnixTime_UTC"][:], unit="s")
            # Back Scatter
            bsc = data["Profile_bsc"][:]

            ds = xr.Dataset()
            ds["z"] = (("z"), alt)
            ds["time"] = (("time"), time)
            ds["x"] = (("x"), [0.0])
            ds["y"] = (("y"), [0.0])

            ds["bsc"] = (("time", "z"), bsc)

            # Attributes
            for k, v in atts.attrs.items():
                if isinstance(v, (list, np.ndarray)) and len(v) > 0:
                    ds.attrs[k] = v[0]
                else:
                    ds.attrs[k] = v

            # Coordinates
            try:
                lat = float(ds.attrs.get("Location_lat", 0.0))
                lon = float(ds.attrs.get("Location_lon", 0.0))
            except (TypeError, ValueError):
                lat, lon = 0.0, 0.0

            ds.coords["latitude"] = (("y", "x"), np.array([[lat]]))
            ds.coords["longitude"] = (("y", "x"), np.array([[lon]]))

            return ds
