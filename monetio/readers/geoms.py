"""GEOMS Reader"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from .base import GriddedReader, register_reader
from .drivers import FileUtility


@register_reader("geoms")
class GEOMSReader(GriddedReader):
    def open_dataset(self, files, rename_all=True, squeeze=True, **kwargs):
        """
        Reads GEOMS format files (HDF4/HDF5).
        """
        file_list = FileUtility.expand_paths(files)

        dsets = []
        for f in file_list:
            ds = open_dataset_geoms(f, rename_all=rename_all, squeeze=squeeze)
            dsets.append(ds)

        if not dsets:
            return xr.Dataset()

        if len(dsets) == 1:
            return dsets[0]
        else:
            return xr.concat(dsets, dim="time")


# -----------------------------------------------------------------------------
# Helper functions ported from monetio/profile/geoms.py
# -----------------------------------------------------------------------------


def open_dataset_geoms(fp, *, rename_all=True, squeeze=True):
    from monetio.util import _import_required

    # Check extension
    # If fp is s3 URL, suffix logic works on string
    ext = Path(fp).suffix.lower()

    fs = FileUtility.get_fs(fp)

    if ext in {".h4", ".hdf4", ".hdf"}:
        pyhdf_SD = _import_required("pyhdf.SD")
        # pyhdf needs a local file usually
        # If remote, download to temp
        if fp.startswith("s3://"):
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=ext, delete=True) as tmp:
                fs.get(fp, tmp.name)
                sd = pyhdf_SD.SD(tmp.name)
                # ... read ...
                data_vars, attrs = _read_hdf4(sd)
                sd.end()
        else:
            sd = pyhdf_SD.SD(str(fp))
            data_vars, attrs = _read_hdf4(sd)
            sd.end()

    elif ext in {".h5", ".he5", ".hdf5"}:
        import h5py

        # h5py works with file-like object from s3fs
        f_obj = fs.open(fp, "rb")
        f = h5py.File(f_obj, "r")

        data_vars = {
            k: (
                tuple(_rename_h5_dim(str(d)) for d in v.dims),
                v[...],
                dict(v.attrs),
            )
            for k, v in f.items()
        }
        attrs = dict(f.attrs)
        f.close()
        f_obj.close()
    else:
        raise ValueError(f"unrecognized file extension: {ext!r}")

    ds = xr.Dataset(
        data_vars=data_vars,
        attrs=attrs,
    )

    instru_coords = [
        "LATITUDE.INSTRUMENT",
        "LONGITUDE.INSTRUMENT",
        "ALTITUDE.INSTRUMENT",
    ]
    for vn in instru_coords:
        if vn in ds:
            da = ds[vn]
            if da.ndim == 0:
                ds = ds.set_coords(vn)
                continue
            (dim_name0,) = da.dims
            dim_name = _rename_var(vn)
            ds = ds.set_coords(vn).rename_dims({dim_name0: dim_name})

    rename_main_dims = {"DATETIME": "time", "ALTITUDE": "altitude"}
    for ref, new_dim in list(rename_main_dims.items()):
        if ref not in ds:
            del rename_main_dims[ref]
            continue
        n = ds[ref].size
        time_dims = [
            dim_name
            for dim_name, dim_size in ds.sizes.items()
            if dim_name.startswith("fakeDim") and dim_size == n
        ]
        ds = ds.rename_dims({dim_name: new_dim for dim_name in time_dims})

    for vn, da in ds.variables.items():
        if da.ndim >= 1 and da.dims[-1].startswith("fakeDim") and da.dtype.kind == "f":
            n = da.sizes[da.dims[-1]]
            if n == 1:
                ds[vn] = da.squeeze(dim=da.dims[-1])

    remaining_vns = [
        vn
        for vn, da in ds.variables.items()
        if any(dim.startswith("fakeDim") for dim in da.dims)
    ]
    for vn in remaining_vns:
        da = ds[vn]
        if not da.dtype.kind == "S":
            continue
        *other_dims, fake_dim = da.dims
        other_dims = other_dims
        if not other_dims:
            ds[vn] = ((), "".join(c.decode("utf-8") for c in da.values), da.attrs)
        else:
            ds[vn] = (
                da.str.decode("utf-8")
                .to_series()
                .groupby(other_dims)
                .agg("".join)
                .to_xarray()
                .astype(str)
                .drop_vars(other_dims)
                .assign_attrs(da.attrs)
            )

    unique_dims = set(ds.dims)
    fake_dims = {dim for dim in unique_dims if dim.startswith("fakeDim")}
    if fake_dims:
        pass  # Warning omitted for brevity

    for vn, da in ds.variables.items():
        if da.dtype.kind == "S":
            ds[vn] = da.astype(str)
        elif da.dtype.kind == "O":
            try:
                x = da.values[0]
            except IndexError:
                x = da.item()
            if isinstance(x, bytes):
                ds[vn] = da.astype(str)
        elif da.dtype.kind == "f":
            if da.dtype.byteorder not in {"=", "|"}:
                ds[vn] = da.astype(da.dtype.newbyteorder("="))

    ds = ds.set_coords(list(rename_main_dims))

    if "DATA_START_DATE" in attrs:
        tstart_from_attr = pd.Timestamp(attrs["DATA_START_DATE"])
        if "DATETIME" in ds:
            t = _dti_from_mjd2000(ds.DATETIME)
            ds["DATETIME"].values = t

    if "DATETIME.START" in ds:
        ds["DATETIME.START"].values = _dti_from_mjd2000(ds["DATETIME.START"])
    if "DATETIME.STOP" in ds:
        ds["DATETIME.STOP"].values = _dti_from_mjd2000(ds["DATETIME.STOP"])

    ds = ds.rename_vars(rename_main_dims)
    ds = ds.rename_vars({old: _rename_var(old) for old in instru_coords if old in ds})

    if rename_all:
        ds = ds.rename_vars({old: _rename_var(old) for old in ds.data_vars})

    if "latitude_instrument" in ds.coords and "latitude" not in ds.coords:
        ds = ds.rename(
            {
                "latitude_instrument": "latitude",
                "longitude_instrument": "longitude",
            }
        )

    if squeeze:
        ds = ds.squeeze()

    return ds


def _read_hdf4(sd):
    data_vars = {}
    for name, _ in sd.datasets().items():
        sds = sd.select(name)
        data = sds.get()
        dims = tuple(sds.dimensions())
        attrs = sds.attributes()
        data_vars[name] = (dims, data, attrs)
        sds.endaccess()
    attrs = sd.attributes()
    return data_vars, attrs


def _rename_h5_dim(s):
    import re

    s_re = r'<"(.*)" dimension (\d+) of HDF5 dataset at (\d+)>'
    m = re.fullmatch(s_re, s)
    if m is None:
        raise ValueError(f"unexpected str of h5 dim: {s!r}.")
    label, num, _ = m.groups()
    return f"fakeDim{num}{label}"


def _rename_var(vn, *, under="_", dot="_"):
    return vn.lower().replace("_", under).replace(".", dot)


def _dti_from_mjd2000(x):
    # assert x.VAR_UNITS == "MJD2K" or x.VAR_UNITS == "MJD2000"
    return pd.to_datetime(np.asarray(x) + 2400000.5 + 51544, unit="D", origin="julian")
