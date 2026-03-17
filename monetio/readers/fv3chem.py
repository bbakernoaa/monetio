"""FV3-CHEM Reader"""

from functools import partial
from typing import Any, List, Union

import numpy as np
import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader
from .sat_utils import update_history


@register_reader("fv3chem")
class FV3ChemReader(GriddedReader):
    """
    Reader for FV3-Chem and AQM model output files (NEMSIO or GRIB2).
    """

    def open_dataset(self, files: Union[str, List[str]], **kwargs: Any) -> xr.Dataset:
        """
        Open a single dataset or multiple files from FV3-Chem outputs.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path, list of paths, or glob pattern.
        **kwargs : Any
            Additional arguments passed to xarray.open_mfdataset or the driver.

        Returns
        -------
        xr.Dataset
            The processed FV3-Chem dataset.
        """
        # Determine file type from the first file (after expansion)
        # We'll use the driver to handle the expansion and opening.
        # But we need to know the type for preprocessing.

        # Let's peek at the first file if possible, or use the pattern.
        # XarrayDriver.open handles both.

        # Setup preprocessing
        if "preprocess" not in kwargs:
            kwargs["preprocess"] = partial(fv3chem_preprocess)

        if "concat_dim" not in kwargs:
            kwargs["concat_dim"] = "time"
        if "combine" not in kwargs:
            kwargs["combine"] = "nested"

        ds = self.driver.open(files, **kwargs)

        # Some post-processing that might depend on the filenames (for NEMSIO time)
        # If it's NEMSIO, we might need to fix the time coordinate.
        # The preprocess function doesn't easily have access to the filename
        # unless we pass it or it's in the dataset attributes.
        # Often open_mfdataset adds 'source' or similar.
        # For now, let's see if we can handle time in preprocess or here.

        # Actually, NEMSIO time fix depends on the filename 'atmfXXX'.
        # If we use open_mfdataset, we can get the filenames from the dataset.
        # In xarray >= 0.20, ds.encoding.get('source') might work if opened individually.
        # But for mfdataset, it's on the data variables.

        if _is_nemsio(ds):
            # Try to fix time if filenames are available in attributes or encoding
            # For mfdataset, we might need to check each file.
            # If the user passed multiple files, we can try to match them.
            # For now, let's assume we can handle it if 'time' is present.
            pass

        ds = self.harmonize(ds)

        # Update history
        ds = update_history(ds, "Read FV3-Chem data.")

        return ds


def fv3chem_preprocess(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess function for a single FV3-Chem file.

    Parameters
    ----------
    ds : xarray.Dataset
        Input FV3-Chem dataset.

    Returns
    -------
    xarray.Dataset
        Processed dataset.
    """
    # 1. Identify type
    nemsio = _is_nemsio(ds)
    grib = _is_grib(ds)

    # 2. Rename variables
    ds = _rename_func(ds, {})

    # 3. Handle specific types
    if nemsio:
        ds = _fix_nemsio(ds)
        # Note: Time fix for NEMSIO often requires the filename.
        # If 'source' is in encoding, we can use it.
        source = ds.encoding.get("source")
        if source:
            ds = _fix_time_nemsio(ds, source)
    elif grib:
        ds = _fix_grib2(ds)

    # 4. Scientific Hygiene: Strip whitespace from all string attributes
    for var in ds.variables:
        for attr, val in ds[var].attrs.items():
            if isinstance(val, str):
                ds[var].attrs[attr] = val.strip()

    # Update history
    ds = update_history(ds, "Preprocessed FV3-Chem data.")

    return ds


def _is_nemsio(ds: xr.Dataset) -> bool:
    """
    Check if the dataset originates from a NEMSIO file.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to check.

    Returns
    -------
    bool
        True if NEMSIO.
    """
    # NEMSIO-derived NetCDF files often have 'hgtsfc' and 'delz'
    # or specific attributes.
    return "hgtsfc" in ds.variables and "delz" in ds.variables


def _is_grib(ds: xr.Dataset) -> bool:
    """
    Check if the dataset originates from a GRIB2 file.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to check.

    Returns
    -------
    bool
        True if GRIB.
    """
    # GRIB2-derived NetCDF files (via fv3grib2nc4) often have long var names
    # or 'lat_0' / 'lon_0'
    return "lat_0" in ds.coords or any("entireatmosphere" in str(v) for v in ds.data_vars)


def _fix_time_nemsio(ds: xr.Dataset, fname: Union[str, List[str]]) -> xr.Dataset:
    """
    Parse and fix time coordinate for NEMSIO-derived NetCDF files.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to fix.
    fname : Union[str, List[str]]
        Filename(s) used to extract forecast hour.

    Returns
    -------
    xarray.Dataset
        The dataset with corrected time.
    """

    def _get_hour(fn: str) -> int:
        try:
            # Look for atmfXXX
            import re

            match = re.search(r"atmf(\d{3})", fn)
            if match:
                return int(match.group(1))
            return 0
        except (IndexError, ValueError):
            return 0

    if isinstance(fname, (list, tuple, np.ndarray)):
        # If we have multiple filenames and multiple times, try to align them
        if "time" in ds.dims and ds.sizes["time"] == len(fname):
            hours = [_get_hour(fn) for fn in fname]
            tdeltas = [pd.Timedelta(h, unit="h") for h in hours]
            tdelta_da = xr.DataArray(tdeltas, dims="time")
            ds["time"] = ds.time + tdelta_da
        elif ds.sizes.get("time", 1) == 1:
            # Single time point in the dataset, use first filename
            hour = _get_hour(fname[0])
            if hour > 0:
                ds["time"] = ds.time + pd.Timedelta(hour, unit="h")
    else:
        # Single file
        hour = _get_hour(str(fname))
        if hour > 0:
            ds["time"] = ds.time + pd.Timedelta(hour, unit="h")

    # Update history
    ds = update_history(ds, "Fixed NEMSIO time coordinate from filename.")

    return ds


def _fix_nemsio(ds: xr.Dataset) -> xr.Dataset:
    """
    Fix NEMSIO-derived NetCDF files by renaming and calculating height.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to fix.

    Returns
    -------
    xarray.Dataset
        The fixed dataset.
    """
    if "hgtsfc" in ds.variables and "delz" in ds.variables:
        ds["geohgt"] = _calc_nemsio_hgt(ds)

    # Update history
    ds = update_history(ds, "Fixed NEMSIO variables and calculated geopotential height.")

    return ds


def _rename_func(ds: xr.Dataset, rename_dict: dict) -> xr.Dataset:
    """
    Rename variables based on patterns and a mapping dictionary.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to rename variables in.
    rename_dict : dict
        Explicit rename mapping.

    Returns
    -------
    xarray.Dataset
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

    # Update history
    if actual_rename:
        ds = update_history(ds, f"Renamed variables: {', '.join(actual_rename.values())}.")

    return ds


def _fix_grib2(ds: xr.Dataset) -> xr.Dataset:
    """
    Fix GRIB2 files by renaming variables and handling coordinates.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to fix.

    Returns
    -------
    xarray.Dataset
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
        lat_name = "y" if "y" in ds.coords else "latitude"
        lon_name = "x" if "x" in ds.coords else "longitude"

        lon_2d, lat_2d = xr.broadcast(ds[lon_name], ds[lat_name])
        ds["longitude"] = lon_2d
        ds["latitude"] = lat_2d
        ds = ds.set_coords(["latitude", "longitude"])

    # Update history
    ds = update_history(ds, "Fixed GRIB2 variables and generated 2D coordinates.")

    return ds


def _calc_nemsio_hgt(ds: xr.Dataset) -> xr.DataArray:
    """
    Calculate geopotential height for NEMSIO-derived NetCDF files.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing surface height and layer thickness.

    Returns
    -------
    xarray.DataArray
        The calculated geopotential height.
    """
    sfc = ds.hgtsfc
    dz = ds.delz
    # z is usually the third dimension (LAY or z)
    z_dim = [d for d in dz.dims if d in ["LAY", "z", "bottom_top"]]
    if not z_dim:
        z_dim = dz.dims[0]
    else:
        z_dim = z_dim[0]

    z = dz.cumsum(dim=z_dim) + sfc
    z.name = "geohgt"
    z.attrs["long_name"] = "Geopotential Height"
    z.attrs["units"] = "m"
    return z
