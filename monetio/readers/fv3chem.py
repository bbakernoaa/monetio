"""FV3-CHEM Reader"""

from datetime import datetime
from typing import Any, List, Tuple, Union

import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader


@register_reader("fv3chem")
class FV3ChemReader(GriddedReader):
    """
    Reader for FV3-CHEM output in NetCDF format (converted from nemsio or grib2).
    """

    def open_dataset(
        self,
        files: Union[str, List[str]],
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Open a single dataset or multiple files from fv3chem outputs.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path(s) or glob pattern.
        **kwargs : dict
            Additional arguments passed to XarrayDriver.open.

        Returns
        -------
        xr.Dataset
            The loaded FV3-CHEM dataset.
        """
        from .drivers import FileUtility

        file_list = FileUtility.expand_paths(files)
        if not file_list:
            raise FileNotFoundError(f"No files found for {files}")

        is_nemsio, is_grib = self._check_file_type(file_list)

        if not is_nemsio and not is_grib:
            raise ValueError("File format not recognized. Ensure 'nemsio' or 'grib2' in filename.")

        # Default merge strategy
        if "concat_dim" not in kwargs:
            kwargs["concat_dim"] = "time"
        if "combine" not in kwargs:
            kwargs["combine"] = "nested"

        ds = self.driver.open(file_list, **kwargs)

        # Post-processing
        if is_nemsio:
            ds = _fix_nemsio(ds)
            ds = _fix_time_nemsio(ds, file_list)
        elif is_grib:
            ds = _fix_grib2(ds)

        # Standard renaming
        rename_dict = {"grid_yt": "y", "grid_xt": "x", "pfull": "z", "phalf": "z_i"}
        rename_dict = {k: v for k, v in rename_dict.items() if k in ds.dims or k in ds.coords}
        ds = ds.rename(rename_dict)

        # Update history
        history = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read FV3-CHEM data."
        if "history" in ds.attrs:
            ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
        else:
            ds.attrs["history"] = history

        return self.harmonize(ds)

    def _check_file_type(self, names: List[str]) -> Tuple[bool, bool]:
        """Identify if files are nemsio-style or grib2-style NetCDF."""
        is_nemsio = any("nemsio" in n.lower() for n in names)
        is_grib = any(x in names[0].lower() for x in ["grb2", "grib2", "grb"])
        return is_nemsio, is_grib


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def _fix_time_nemsio(ds: xr.Dataset, filenames: List[str]) -> xr.Dataset:
    """Reconstruct time coordinate for nemsio-converted files."""
    if "time" not in ds.coords:
        return ds

    def _get_hour(fn: str) -> int:
        try:
            parts = fn.split(".")
            atmf = [p for p in parts if "atmf" in p.lower()][0]
            return int(atmf.lower().replace("atmf", ""))
        except (IndexError, ValueError):
            return 0

    if ds.sizes["time"] == len(filenames):
        # One time per file
        forecast_hours = [_get_hour(f) for f in filenames]

        # To maintain laziness, we use apply_ufunc if possible, but for a small list of filenames,
        # computing only the time coordinate is generally acceptable and safer than complex gufunc.
        # However, to be strict Aero, we'll try to avoid ds.time.values.
        base_times = ds.time
        # Create a DataArray for forecast hours
        fh_da = xr.DataArray(forecast_hours, dims=["time"], coords={"time": base_times})

        def _add_hours(t, h):
            # Vectorized addition of hours to timestamps
            # Works with numpy datetime64
            return t + h.astype("timedelta64[h]")

        new_times = xr.apply_ufunc(_add_hours, base_times, fh_da, dask="parallelized")
        ds = ds.assign_coords(time=new_times)
    else:
        # Single file or mismatch
        hour = _get_hour(filenames[0])
        if hour > 0:
            ds = ds.assign_coords(time=ds.time + pd.Timedelta(hour, unit="h"))

    return ds


def _fix_nemsio(ds: xr.Dataset) -> xr.Dataset:
    """Standardize nemsio variables and calculate height."""
    rename_map = {}
    for var in ds.data_vars:
        if "midlayer" in var:
            rename_map[var] = var.replace("midlayer", "")

    if "pp25" in ds.data_vars:
        rename_map["pp25"] = "pm25"
    if "pp10" in ds.data_vars:
        rename_map["pp10"] = "pm10"

    ds = ds.rename({k: v for k, v in rename_map.items() if k in ds.data_vars})

    if "delz" in ds.data_vars and "hgtsfc" in ds.data_vars:
        z_dim = next((d for d in ["pfull", "z", "layer"] if d in ds.dims), None)
        if z_dim:
            ds["geohgt"] = ds.delz.cumsum(dim=z_dim) + ds.hgtsfc
            ds.geohgt.attrs.update({"units": "m", "long_name": "Geopotential Height"})

    return ds


def _fix_grib2(ds: xr.Dataset) -> xr.Dataset:
    """Standardize grib2 variables and grid."""
    name_map = {
        "AOTK_aerosol_EQ_Total_Aerosol_aerosol_size_LT_2eM05_aerosol_wavelength_GE_5D45eM07_LE_5D65eM07_entireatmosphere": "pm25aod550",
        "PMTF_chemical_Total_Aerosol_aerosol_size__2_5e_06_aerosol_wavelength_____code_table_4_91_255_surface": "sfc_pm25",
        "PMTC_aerosol_EQ_Total_Aerosol_aerosol_size_LT_1eM05_surface": "sfc_pm10",
    }
    for var in ds.data_vars:
        if "AOTK_chemical_Total_Aerosol" in var and "5_45e_07_5_65e_07" in var:
            name_map[var] = "pm25aod550"

    ds = ds.rename({k: v for k, v in name_map.items() if k in ds.data_vars})

    if "latitude" not in ds.coords:
        if "lat_0" in ds.coords:
            ds = ds.rename({"lat_0": "latitude", "lon_0": "longitude"})
        elif "lat" in ds.coords:
            ds = ds.rename({"lat": "latitude", "lon": "longitude"})

    if "latitude" in ds.coords and ds.latitude.ndim == 1:
        lon_2d, lat_2d = xr.broadcast(ds.longitude, ds.latitude)
        y_dim = ds.latitude.dims[0]
        x_dim = ds.longitude.dims[0]
        ds = ds.assign_coords(
            latitude=lat_2d.transpose(y_dim, x_dim),
            longitude=lon_2d.transpose(y_dim, x_dim),
        )

    return ds
