"""FV3-CHEM Reader"""

import datetime
from glob import glob
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from numpy import sort

from .base import GriddedReader, register_reader


@register_reader("fv3chem")
class FV3ChemReader(GriddedReader):
    def open_dataset(self, files: Union[str, List[str]], **kwargs: Optional[dict]) -> xr.Dataset:
        """
        Open a single dataset or multiple files from fv3chem outputs (nemsio or grib2).

        Parameters
        ----------
        files : Union[str, List[str]]
            File path, list of paths, or glob pattern.
        **kwargs : dict
            Additional arguments passed to the driver and driver.open.

        Returns
        -------
        xr.Dataset
            The processed FV3-Chem dataset.
        """
        # We manually handle file expansion here because of the nemsio/grib logic
        # However, the driver also expands paths.
        # Let's use the driver's capability to expand paths first.

        # But we need to know if it is nemsio or grib before calling open_mfdataset/open_dataset
        # because the arguments differ (engine="pynio" vs "nemsio"? No, nemsio uses standard open but needs preprocessing).

        # Actually, original code uses `xr.open_dataset` for nemsio (netcdf-like?) or grib.
        # It seems `nemsio` might be opened as netcdf if converted?
        # Original docstring says: "must preprocess the files with nemsio2nc4 or fv3grib2nc4"
        # So they are actually NetCDF files.

        # Let's inspect the files to decide.
        if isinstance(files, str):
            expanded_files = sort(glob(files))
        else:
            expanded_files = sort(files)

        if len(expanded_files) == 0:
            raise FileNotFoundError(f"No files found for {files}")

        names, nemsio, grib = self._check_file_type(expanded_files)

        if not nemsio and not grib:
            # Fallback or error
            # Original code raises ValueError
            raise ValueError("File format not recognized. Ensure nemsio or grib2/grb2 in filename.")

        # Prepare kwargs
        if "concat_dim" not in kwargs:
            kwargs["concat_dim"] = "time"
        if "combine" not in kwargs:
            kwargs["combine"] = "nested"  # Default for mfdataset usually

        # Open
        ds = self.driver.open(names, **kwargs)

        # Post-process
        if nemsio:
            ds = _fix_nemsio(ds)
            # Fix time for nemsio needs filename(s)
            ds = _fix_time_nemsio(ds, names)
        elif grib:
            ds = _fix_grib2(ds)

        ds = self.harmonize(ds)

        # Update history
        history = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read FV3-Chem data."
        if "history" in ds.attrs:
            ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
        else:
            ds.attrs["history"] = history

        return ds

    def _check_file_type(self, names: List[str]) -> tuple:
        """
        Check if files are nemsio or grib2 format.

        Parameters
        ----------
        names : List[str]
            List of filenames.

        Returns
        -------
        tuple
            (names, nemsio_flag, grib_flag)
        """
        nemsios = [True for i in names if "nemsio" in i]
        gribs = [True for i in names if "grb2" in i or "grib2" in i or "grb" in i]
        grib = False
        nemsio = False
        if len(nemsios) >= 1:
            nemsio = True
        elif len(gribs) >= 1:
            grib = True
        return names, nemsio, grib


# -----------------------------------------------------------------------------
# Helper functions ported from monetio/models/fv3chem.py
# -----------------------------------------------------------------------------


def _fix_time_nemsio(ds: xr.Dataset, fname: Union[str, List[str]]) -> xr.Dataset:
    """
    Parse and fix time coordinate for NEMSIO-derived NetCDF files.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to fix.
    fname : Union[str, List[str]]
        Filename(s) used to extract forecast hour.

    Returns
    -------
    xr.Dataset
        The dataset with corrected time.
    """
    # If fname is a list, we handle it.
    is_multi = isinstance(fname, (list, tuple, np.ndarray)) and len(fname) > 1

    # Extract hour from filename(s)
    # "atmf" in i -> "...atmf003..." -> hour = 3.
    def _get_hour(fn):
        try:
            hour_str = [i for i in fn.split(".") if "atmf" in i][0][-3:]
            return int(hour_str)
        except (IndexError, ValueError):
            return 0

    if is_multi:
        # If ds.time matches fname length (one time per file)
        if "time" in ds.dims and ds.sizes["time"] == len(fname):
            hours = [_get_hour(fn) for fn in fname]
            tdeltas = [pd.Timedelta(h, unit="h") for h in hours]

            # Use xarray arithmetic to add timedeltas lazily
            # If we have multiple files with different offsets,
            # we can create a DataArray of timedeltas and add it.
            tdelta_da = xr.DataArray(tdeltas, dims="time")
            ds["time"] = ds.time + tdelta_da
    else:
        # Single file
        fn = fname[0] if isinstance(fname, (list, tuple)) else fname
        hour = _get_hour(fn)
        if hour > 0:
            ds["time"] = ds.time + pd.Timedelta(hour, unit="h")

    return ds


def _fix_nemsio(ds: xr.Dataset) -> xr.Dataset:
    """
    Fix NEMSIO-derived NetCDF files by renaming and calculating height.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to fix.

    Returns
    -------
    xr.Dataset
        The fixed dataset.
    """
    ds = _rename_func(ds, {})
    try:
        if "hgtsfc" in ds.variables and "delz" in ds.variables:
            ds["geohgt"] = _calc_nemsio_hgt(ds)
    except Exception:
        pass
    return ds


def _rename_func(ds: xr.Dataset, rename_dict: dict) -> xr.Dataset:
    """
    Rename variables based on patterns and a mapping dictionary.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to rename variables in.
    rename_dict : dict
        Explicit rename mapping.

    Returns
    -------
    xr.Dataset
        The dataset with renamed variables.
    """
    final_dict = rename_dict.copy()
    # Pattern based renaming: '...midlayer' -> '...'
    pattern_renames = {i: i.split("midlayer")[0] for i in ds.data_vars if "midlayer" in i}
    final_dict.update(pattern_renames)

    # Filter to only existing variables
    actual_rename = {k: v for k, v in final_dict.items() if k in ds.variables}

    # Add specific ones
    if "pp25" in ds.variables:
        actual_rename["pp25"] = "pm25"
    if "pp10" in ds.variables:
        actual_rename["pp10"] = "pm10"

    if actual_rename:
        ds = ds.rename(actual_rename)

    return ds


def _fix_grib2(ds: xr.Dataset) -> xr.Dataset:
    """
    Fix GRIB2 files by renaming variables and handling coordinates.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to fix.

    Returns
    -------
    xr.Dataset
        The fixed dataset.
    """
    rename_dict = {
        "AOTK_aerosol_EQ_Total_Aerosol_aerosol_size_LT_2eM05_aerosol_wavelength_GE_5D45eM07_LE_5D65eM07_entireatmosphere": "pm25aod550",
        "AOTK_aerosol_EQ_Dust_Dry_aerosol_size_LT_2eM05_aerosol_wavelength_GE_5D45eM07_LE_5D65eM07_entireatmosphere": "dust25aod550",
        "AOTK_chemical_Dust_Dry_aerosol_size__2e_05_aerosol_wavelength_5_45e_07_5_65e_07_entireatmosphere": "dust25aod550",
        "AOTK_aerosol_EQ_Sea_Salt_Dry_aerosol_size_LT_2eM05_aerosol_wavelength_GE_5D45eM07_LE_5D65eM07_entireatmosphere": "salt25aod550",
        "AOTK_chemical_Sea_Salt_Dry_aerosol_size__2e_05_aerosol_wavelength_5_45e_07_5_65e_07_entireatmosphere": "salt25aod550",
        "AOTK_aerosol_EQ_Sulphate_Dry_aerosol_size_LT_2eM05_aerosol_wavelength_GE_5D45eM07_LE_5D65eM07_entireatmosphere": "sulf25aod550",
        "AOTK_chemical_Sulphate_Dry_aerosol_size__2e_05_aerosol_wavelength_5_45e_07_5_65e_07_entireatmosphere": "sulf25aod550",
        "AOTK_aerosol_EQ_Particulate_Organic_Matter_Dry_aerosol_size_LT_2eM05_aerosol_wavelength_GE_5D45eM07_LE_5D65eM07_entireatmosphere": "oc25aod550",
        "AOTK_chemical_Particulate_Organic_Matter_Dry_aerosol_size__2e_05_aerosol_wavelength_5_45e_07_5_65e_07_entireatmosphere": "oc25aod550",
        "AOTK_aerosol_EQ_Black_Carbon_Dry_aerosol_size_LT_2eM05_aerosol_wavelength_GE_5D45eM07_LE_5D65eM07_entireatmosphere": "bc25aod550",
        "AOTK_chemical_Black_Carbon_Dry_aerosol_size__2e_05_aerosol_wavelength_5_45e_07_5_65e_07_entireatmosphere": "bc25aod550",
        "COLMD_aerosol_EQ_Total_Aerosol_aerosol_size_LT_1eM05_entireatmosphere": "tc_aero10",
        "COLMD_aerosol_EQ_Total_Aerosol_aerosol_size_LT_2D5eM06_entireatmosphere": "tc_aero25",
        "COLMD_aerosol_EQ_Dust_Dry_aerosol_size_LT_2D5eM06_entireatmosphere": "tc_dust25",
        "COLMD_chemical_Dust_Dry_aerosol_size__2_5e_06_aerosol_wavelength_____code_table_4_91_255_entireatmosphere": "tc_dust25",
        "COLMD_aerosol_EQ_Sea_Salt_Dry_aerosol_size_LT_2D5eM06_entireatmosphere": "tc_salt25",
        "COLMD_chemical_Sea_Salt_Dry_aerosol_size__2_5e_06_aerosol_wavelength_____code_table_4_91_255_entireatmosphere": "tc_salt25",
        "COLMD_aerosol_EQ_Black_Carbon_Dry_aerosol_size_LT_2D36eM08_entireatmosphere": "tc_bc236",
        "COLMD_aerosol_EQ_Particulate_Organic_Matter_Dry_aerosol_size_LT_4D24eM08_entireatmosphere": "tc_oc424",
        "COLMD_aerosol_EQ_Sulphate_Dry_aerosol_size_LT_2D5eM06_entireatmosphere": "tc_sulf25",
        "COLMD_chemical_Sulphate_Dry_aerosol_size__2_5e_06_aerosol_wavelength_____code_table_4_91_255_entireatmosphere": "tc_sulf25",
        "PMTF_chemical_Dust_Dry_aerosol_size__2_5e_06_aerosol_wavelength_____code_table_4_91_255_surface": "sfc_dust25",
        "PMTF_chemical_Sea_Salt_Dry_aerosol_size__2_5e_06_aerosol_wavelength_____code_table_4_91_255_surface": "sfc_salt25",
        "PMTF_chemical_Total_Aerosol_aerosol_size__2_5e_06_aerosol_wavelength_____code_table_4_91_255_surface": "sfc_pm25",
        "PMTF_aerosol_EQ_Total_Aerosol_aerosol_size_LT_2D5eM06_surface": "sfc_pm25",
        "PMTC_aerosol_EQ_Total_Aerosol_aerosol_size_LT_1eM05_surface": "sfc_pm10",
        "PMTF_aerosol_EQ_Sea_Salt_Dry_aerosol_size_LT_2D5eM06_surface": "sfc_salt25",
        "PMTF_aerosol_EQ_Dust_Dry_aerosol_size_LT_2D5eM06_surface": "sfc_dust25",
        "PMTF_chemical_Dust_Dry_aerosol_size___2e_07__2e_06_aerosol_wavelength_____code_table_4_91_255_1hybridlevel": "dustmr1p1",
        "PMTF_chemical_Dust_Dry_aerosol_size___2e_06__3_6e_06_aerosol_wavelength_____code_table_4_91_255_1hybridlevel": "dustmr2p5",
        "PMTC_chemical_Dust_Dry_aerosol_size___3_6e_06__6e_06_aerosol_wavelength_____code_table_4_91_255_1hybridlevel": "dustmr4p8",
        "PMTC_chemical_Dust_Dry_aerosol_size___6e_06__1_2e_05_aerosol_wavelength_____code_table_4_91_255_1hybridlevel": "dustmr9p0",
        "PMTC_chemical_Dust_Dry_aerosol_size___1_2e_05__2e_05_aerosol_wavelength_____code_table_4_91_255_1hybridlevel": "dustmr16p0",
        "PMTF_chemical_Sea_Salt_Dry_aerosol_size___2e_07__1e_06_aerosol_wavelength_____code_table_4_91_255_1hybridlevel": "saltmr0p6",
        "PMTC_chemical_Sea_Salt_Dry_aerosol_size___1e_06__3e_06_aerosol_wavelength_____code_table_4_91_255_1hybridlevel": "saltmr2p0",
        "PMTC_chemical_Sea_Salt_Dry_aerosol_size___3e_06__1e_05_aerosol_wavelength_____code_table_4_91_255_1hybridlevel": "saltmr6p5",
        "PMTC_chemical_Sea_Salt_Dry_aerosol_size___1e_05__2e_05_aerosol_wavelength_____code_table_4_91_255_1hybridlevel": "saltmr10p5",
        "PMTF_chemical_Sulphate_Dry_aerosol_size__1_39e_07_aerosol_wavelength_____code_table_4_91_255_1hybridlevel": "sulfmr1p36",
        "PMTF_chemical_chemical_62016_aerosol_size__4_24e_08_aerosol_wavelength_____code_table_4_91_255_1hybridlevel": "aero1_mr0p0424",
        "PMTF_chemical_chemical_62015_aerosol_size__4_24e_08_aerosol_wavelength_____code_table_4_91_255_1hybridlevel": "aero2_mr0p0424",
        "PMTF_chemical_chemical_62014_aerosol_size__2_36e_08_aerosol_wavelength_____code_table_4_91_255_1hybridlevel": "aero1_mr0p0236",
        "PMTF_chemical_chemical_62013_aerosol_size__2_36e_08_aerosol_wavelength_____code_table_4_91_255_1hybridlevel": "aero2_mr0p0236",
        "level": "z",
        "AOTK_chemical_Total_Aerosol_aerosol_size__2e_05_aerosol_wavelength_3_38e_07_3_42e_07_entireatmosphere": "pm25aod340",
        "ASYSFK_chemical_Total_Aerosol_aerosol_size__2e_05_aerosol_wavelength_3_38e_07_3_42e_07_entireatmosphere": "AF_pm25aod340",
        "SSALBK_chemical_Total_Aerosol_aerosol_size__2e_05_aerosol_wavelength_3_38e_07_3_42e_07_entireatmosphere": "ssa_pm25aod340",
        "AOTK_chemical_Total_Aerosol_aerosol_size__2e_05_aerosol_wavelength_4_3e_07_4_5e_07_entireatmosphere": "pm25aod440",
        "AOTK_chemical_Total_Aerosol_aerosol_size__2e_05_aerosol_wavelength_5_45e_07_5_65e_07_entireatmosphere": "pm25aod550",
        "var0_20_112_chemical_Total_Aerosol_aerosol_size__2e_05_aerosol_wavelength_5_45e_07_5_65e_07_entireatmosphere": "tc_pm25aod550",
        "var0_20_112_chemical_Dust_Dry_aerosol_size__2e_05_aerosol_wavelength_5_45e_07_5_65e_07_entireatmosphere": "tc_dust25aod550",
        "var0_20_112_chemical_Sea_Salt_Dry_aerosol_size__2e_05_aerosol_wavelength_5_45e_07_5_65e_07_entireatmosphere": "tc_salt25aod550",
        "var0_20_112_chemical_Sulphate_Dry_aerosol_size__7e_07_aerosol_wavelength_5_45e_07_5_65e_07_entireatmosphere": "tc_sulf25aod550",
        "var0_20_112_chemical_Particulate_Organic_Matter_Dry_aerosol_size__7e_07_aerosol_wavelength_5_45e_07_5_65e_07_entireatmosphere": "tc_sulfaod550",
        "var0_20_112_chemical_Black_Carbon_Dry_aerosol_size__7e_07_aerosol_wavelength_5_45e_07_5_65e_07_entireatmosphere": "tc_ocaod550",
        "AOTK_chemical_Total_Aerosol_aerosol_size__2e_05_aerosol_wavelength_6_2e_07_6_7e_07_entireatmosphere": "pm25aod640",
        "AOTK_chemical_Total_Aerosol_aerosol_size__2e_05_aerosol_wavelength_8_41e_07_8_76e_07_entireatmosphere": "pm25aod860",
        "AOTK_chemical_Total_Aerosol_aerosol_size__2e_05_aerosol_wavelength_1_628e_06_1_652e_06_entireatmosphere": "pm25aod1645",
        "AOTK_chemical_Total_Aerosol_aerosol_size__2e_05_aerosol_wavelength_1_1e_05_1_12e_05_entireatmosphere": "pm25aod11500",
        "AOTK_aerosol_EQ_Total_Aerosol_aerosol_size_LT_2eM05_aerosol_wavelength_GE_3D38eM07_LE_3D42eM07_entireatmosphere": "pm25aod340_eq",
        "SSA_pm25aod340": "SSA_pm25aod340",
        "AOTK_aerosol_EQ_Total_Aerosol_aerosol_size_LT_2eM05_aerosol_wavelength_GE_4D3eM07_LE_4D5eM07_entireatmosphere": "pm25aod440",
        "SCTAOTK_aerosol_EQ_Total_Aerosol_aerosol_size_LT_2eM05_aerosol_wavelength_GE_5D45eM07_LE_5D65eM07_entireatmosphere": "SA_pm25aod550",
        "SCTAOTK_aerosol_EQ_Dust_Dry_aerosol_size_LT_2eM05_aerosol_wavelength_GE_5D45eM07_LE_5D65eM07_entireatmosphere": "SA_dust25aod550",
        "SCTAOTK_aerosol_EQ_Sea_Salt_Dry_aerosol_size_LT_2eM05_aerosol_wavelength_GE_5D45eM07_LE_5D65eM07_entireatmosphere": "SA_salt25aod550",
        "SCTAOTK_aerosol_EQ_Sulphate_Dry_aerosol_size_LT_7eM07_aerosol_wavelength_GE_5D45eM07_LE_5D65eM07_entireatmosphere": "SA_sulf07aod550",
        "SCTAOTK_aerosol_EQ_Particulate_Organic_Matter_Dry_aerosol_size_LT_7eM07_aerosol_wavelength_GE_5D45eM07_LE_5D65eM07_entireatmosphere": "SA_oc07aod550",
        "SCTAOTK_aerosol_EQ_Black_Carbon_Dry_aerosol_size_LT_7eM07_aerosol_wavelength_GE_5D45eM07_LE_5D65eM07_entireatmosphere": "SC_bc07aod550",
        "AOTK_aerosol_EQ_Total_Aerosol_aerosol_size_LT_2eM05_aerosol_wavelength_GE_6D2eM07_LE_6D7eM07_entireatmosphere": "pm25aod645",
        "AOTK_aerosol_EQ_Total_Aerosol_aerosol_size_LT_2eM05_aerosol_wavelength_GE_8D41eM07_LE_8D76eM07_entireatmosphere": "pm25aod841",
        "AOTK_aerosol_EQ_Total_Aerosol_aerosol_size_LT_2eM05_aerosol_wavelength_GE_1D628eM06_LE_1D652eM06_entireatmosphere": "pm25aod1628",
        "AOTK_aerosol_EQ_Total_Aerosol_aerosol_size_LT_2eM05_aerosol_wavelength_GE_1D1eM05_LE_1D12eM05_entireatmosphere": "pm25aod11000",
    }

    ds = _rename_func(ds, rename_dict)

    # Check if 'latitude'/'longitude' exist already or need renaming from lat_0/lon_0
    # Grib2 often has lat_0, lon_0
    rename_coords = {}
    if "latitude" not in ds.coords:
        if "lat_0" in ds.coords:
            rename_coords["lat_0"] = "latitude"
            rename_coords["lon_0"] = "longitude"
        elif "lat" in ds.coords:
            rename_coords["lat"] = "latitude"
            rename_coords["lon"] = "longitude"

    if rename_coords:
        ds = ds.rename(rename_coords)

    # Create 2D grid if 1D (Meshgrid)
    if (
        "latitude" in ds.coords
        and "longitude" in ds.coords
        and ds.latitude.ndim == 1
        and ds.longitude.ndim == 1
    ):
        # Rename dims to y, x if not present
        rename_dims = {}
        if "lat_0" in ds.dims:
            rename_dims["lat_0"] = "y"
            rename_dims["lon_0"] = "x"
        elif "latitude" in ds.dims:
            rename_dims["latitude"] = "y"
            rename_dims["longitude"] = "x"

        if rename_dims:
            ds = ds.rename(rename_dims)

        # Use broadcast to create 2D arrays lazily
        # After renaming dims, coords might also be renamed if they shared the name
        lat_name = "y" if "y" in ds.coords else "latitude"
        lon_name = "x" if "x" in ds.coords else "longitude"

        lon_2d, lat_2d = xr.broadcast(ds[lon_name], ds[lat_name])
        ds["longitude"] = lon_2d
        ds["latitude"] = lat_2d
        ds = ds.set_coords(["latitude", "longitude"])

    return ds


def _calc_nemsio_hgt(ds: xr.Dataset) -> xr.DataArray:
    """
    Calculate geopotential height for NEMSIO-derived NetCDF files.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing surface height and layer thickness.

    Returns
    -------
    xr.DataArray
        The calculated geopotential height.
    """
    sfc = ds.hgtsfc
    dz = ds.delz
    # Level 0 height = sfc + dz[0]
    # Level 1 height = Level 0 + dz[1] = sfc + dz[0] + dz[1]
    # Correct lazy logic: sfc + dz.cumsum()
    z = dz.cumsum(dim="z") + sfc
    z.name = "geohgt"
    z.attrs["long_name"] = "Geopotential Height"
    z.attrs["units"] = "m"
    return z
