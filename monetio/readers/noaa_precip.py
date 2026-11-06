"""NOAA verification precipitation and snowfall grid readers.

Readers accept local paths, glob patterns, lists of paths, and URLs understood
by the shared Xarray driver. They preserve native values and attributes so the
product's units and accumulation definition remain explicit.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np
import xarray as xr

from .base import GriddedReader, register_reader
from .errors import OptionUnsupportedError
from .sat_utils import update_history


def _harmonize_grid(ds: xr.Dataset, aliases: dict[str, str]) -> xr.Dataset:
    """Normalize common coordinate and product variable spellings."""
    rename = {}
    for source, target in aliases.items():
        if source in ds.data_vars and target not in ds.data_vars:
            rename[source] = target
    if rename:
        ds = ds.rename(rename)
    latitude = next(
        (name for name in ("latitude", "lat", "LAT", "Latitude") if name in ds.variables), None
    )
    longitude = next(
        (name for name in ("longitude", "lon", "LON", "Longitude") if name in ds.variables), None
    )
    dim_rename = {}
    reserved_dims = set(ds.dims)
    y_names = {"lat", "latitude", "rows", "scanline", "nlat", "nscan"}
    x_names = {"lon", "longitude", "columns", "ground_pixel", "nlon", "nstep"}
    for dim in ds.dims:
        lowered = dim.lower()
        if lowered in y_names and dim != "y" and "y" not in reserved_dims:
            dim_rename[dim] = "y"
            reserved_dims.add("y")
        elif lowered in x_names and dim != "x" and "x" not in reserved_dims:
            dim_rename[dim] = "x"
            reserved_dims.add("x")
    if dim_rename:
        ds = ds.rename_dims(dim_rename)
    variable_rename = {}
    if latitude and latitude != "latitude" and "latitude" not in ds.variables:
        variable_rename[latitude] = "latitude"
    if longitude and longitude != "longitude" and "longitude" not in ds.variables:
        variable_rename[longitude] = "longitude"
    if variable_rename:
        ds = ds.rename_vars(variable_rename)
    if "time" not in ds.variables:
        if "valid_time" in ds.variables:
            ds = ds.rename_vars({"valid_time": "time"})
        else:
            time_name = next((name for name in ds.variables if name.lower() == "time"), None)
            if time_name:
                ds = ds.rename_vars({time_name: "time"})
    for name, units, standard_name in (
        ("latitude", "degrees_north", "latitude"),
        ("longitude", "degrees_east", "longitude"),
    ):
        if name in ds.variables:
            ds = ds.set_coords(name)
            ds[name].attrs.setdefault("units", units)
            ds[name].attrs.setdefault("standard_name", standard_name)
    if "time" in ds.variables:
        ds = ds.set_coords("time")
    # Keep projection/grid mapping variables and accumulation bounds intact.
    return ds


class _NOAAPrecipReader(GriddedReader):
    """Shared Xarray-driver behavior for NOAA gridded verification products."""

    source_name = ""
    aliases: dict[str, str] = {}

    def open_dataset(self, files: str | list[str], **kwargs) -> xr.Dataset:
        """Open local or directly addressed product files and harmonize names.

        Parameters
        ----------
        files : str or list[str]
            Local files, glob pattern, URLs, or object-store paths accepted by
            MONETIO's Xarray driver.
        **kwargs : dict
            Standard gridded reader and Xarray driver options.

        Returns
        -------
        xarray.Dataset
            Product data with standard time/latitude/longitude coordinate names
            where those coordinates are present. Native units, projections,
            validity bounds, and product attributes are retained.
        """
        ds = super().open_dataset(files, **kwargs)
        ds = _harmonize_grid(ds, self.aliases)
        ds.attrs.setdefault("source", self.source_name)
        return update_history(ds, f"Read {self.source_name} gridded data.")


@register_reader("prism")
class PRISMReader(_NOAAPrecipReader):
    """Read local or directly addressed PRISM daily gridded products."""

    source_name = "PRISM"
    aliases = {
        "ppt": "precipitation_amount",
        "precip": "precipitation_amount",
        "tmin": "minimum_air_temperature",
        "tmax": "maximum_air_temperature",
        "tmean": "mean_air_temperature",
    }

    def open_dataset(self, files: str | list[str], **kwargs) -> xr.Dataset:
        """Open PRISM NetCDF inputs or native single-band BIL raster files.

        BIL inputs must be extracted and have a companion ESRI ``.hdr`` file;
        an optional ``.prj`` file supplies the projected CRS.
        """
        items = [files] if isinstance(files, str | os.PathLike) else list(files)
        bil = [os.fspath(path).lower().endswith(".bil") for path in items]
        if not any(bil):
            return super().open_dataset(files, **kwargs)
        if not all(bil):
            raise ValueError("PRISM inputs cannot mix BIL rasters with other file formats")
        if (
            kwargs.get("use_dask")
            or kwargs.get("defer")
            or kwargs.get("lazy")
            or "chunks" in kwargs
            or kwargs.get("use_virtualizarr")
            or kwargs.get("use_icechunk")
        ):
            raise OptionUnsupportedError(
                "Dask/chunk/virtualization options",
                "prism",
                cause="native BIL reads are eager and do not use virtualization",
            )
        paths = []
        for item in items:
            path = os.fspath(item)
            if "*" in path or "?" in path or "[" in path:
                paths.extend(sorted(glob.glob(path)))
            else:
                paths.append(path)
        if not paths:
            raise FileNotFoundError(f"No PRISM BIL files matched {files!r}")
        datasets = [_read_prism_bil(path) for path in paths]
        ds = datasets[0] if len(datasets) == 1 else xr.concat(datasets, dim="time")
        ds = _harmonize_grid(ds, self.aliases)
        ds.attrs.setdefault("source", self.source_name)
        return update_history(ds, "Read PRISM native BIL data.")


def _read_prism_bil(path: str) -> xr.Dataset:
    """Read one extracted PRISM daily BIL grid and its sidecar metadata."""
    if path.startswith(("http://", "https://", "s3://")):
        raise ValueError("PRISM BIL files must be local and accompanied by their .hdr sidecar")
    raster = Path(path)
    header_path = raster.with_suffix(".hdr")
    if not raster.is_file():
        raise FileNotFoundError(path)
    if not header_path.is_file():
        raise FileNotFoundError(f"PRISM BIL header not found: {header_path}")
    header = {}
    for line in header_path.read_text().splitlines():
        fields = line.split(None, 1)
        if len(fields) == 2:
            header[fields[0].upper()] = fields[1].strip()
    required = ("NROWS", "NCOLS", "NBANDS", "NBITS", "ULXMAP", "ULYMAP", "XDIM", "YDIM")
    missing = [key for key in required if key not in header]
    if missing:
        raise ValueError(f"PRISM BIL header {header_path} is missing {missing}")
    rows, cols = int(header["NROWS"]), int(header["NCOLS"])
    if int(header["NBANDS"]) != 1 or int(header["NBITS"]) != 16:
        raise ValueError("PRISM BIL reader supports single-band 16-bit rasters")
    byteorder = header.get("BYTEORDER", "I").upper()
    dtype = np.dtype("<i2" if byteorder == "I" else ">i2")
    row_bytes = int(header.get("BANDROWBYTES", cols * dtype.itemsize))
    raw = np.fromfile(raster, dtype=np.uint8)
    if raw.size != rows * row_bytes:
        raise ValueError(f"Unexpected byte count in PRISM BIL file {raster}")
    values = (
        raw.reshape(rows, row_bytes)[:, : cols * dtype.itemsize]
        .copy()
        .view(dtype)
        .reshape(rows, cols)
    )
    values = values.astype("float32")
    nodata = float(header.get("NODATA", header.get("NODATA_VALUE", "-9999")))
    values[values == nodata] = np.nan

    match = re.search(r"PRISM_(ppt|tmin|tmax|tmean).*?(\d{8})", raster.name, re.IGNORECASE)
    variable = match.group(1).lower() if match else "value"
    if variable in {"ppt", "tmin", "tmax", "tmean"}:
        values *= 0.01
    x = float(header["ULXMAP"]) + np.arange(cols) * float(header["XDIM"])
    y = float(header["ULYMAP"]) - np.arange(rows) * float(header["YDIM"])
    coords = {"x": x, "y": y}
    data = xr.DataArray(values, dims=("y", "x"), coords=coords, name=variable)
    data.attrs["units"] = (
        "mm" if variable == "ppt" else ("degC" if variable.startswith("t") else "")
    )
    data.attrs["_FillValue"] = nodata
    ds = data.to_dataset()
    if match:
        date = match.group(2)
        ds = ds.expand_dims(time=[np.datetime64(f"{date[:4]}-{date[4:6]}-{date[6:]}", "D")])
    prj = raster.with_suffix(".prj")
    if prj.is_file():
        from pyproj import CRS, Transformer

        crs = CRS.from_wkt(prj.read_text())
        ds["spatial_ref"] = xr.DataArray(
            0, attrs={"crs_wkt": crs.to_wkt(), "spatial_ref": crs.to_wkt()}
        )
        longitude, latitude = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(
            *np.meshgrid(x, y)
        )
        ds = ds.assign_coords(latitude=(("y", "x"), latitude), longitude=(("y", "x"), longitude))
    return ds


@register_reader("ccpa")
class CCPAReader(_NOAAPrecipReader):
    """Read local or directly addressed CCPA precipitation analyses."""

    source_name = "CCPA"
    aliases = {
        "APCP_surface": "precipitation_amount",
        "APCP": "precipitation_amount",
        "precip": "precipitation_amount",
        "precipitation": "precipitation_amount",
    }


@register_reader("nohrsc")
class NOHRSCReader(_NOAAPrecipReader):
    """Read local or directly addressed NOHRSC snowfall analyses."""

    source_name = "NOHRSC Snowfall Analyses"
    aliases = {
        "ASNOW": "snowfall_amount",
        "SNOWFALL": "snowfall_amount",
        "snowfall": "snowfall_amount",
        "WEASD": "snow_water_equivalent",
    }
