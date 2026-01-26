"""FV3-CHEM Reader"""

from glob import glob

from numpy import sort
from pandas import Timedelta, to_datetime

from .base import GriddedReader, register_reader


@register_reader("fv3chem")
class FV3ChemReader(GriddedReader):
    def open_dataset(self, files, **kwargs):
        """
        Open a single dataset or multiple files from fv3chem outputs (nemsio or grib2).
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

        return self.harmonize(ds)

    def _check_file_type(self, names):
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


def _fix_time_nemsio(f, fname):
    time = None
    # If fname is a list, we handle it.
    is_multi = isinstance(fname, (list, tuple, np.ndarray)) and len(fname) > 1

    if "time" in f.coords and f.time.size > 1 and is_multi:
        # This logic seems to assume one time per file usually?
        # Original code: zip(f.time.to_index(), fname)
        # This implies f.time length equals fname length?
        # If open_mfdataset concatenated them, f.time has all times.
        # We need to be careful.
        pass

    # Re-implementing logic carefully.
    # The original logic extracted hour from filename and added to time index?
    # Or replaced time index?

    # "atmf" in i -> "...atmf003..." -> hour = 3.

    # If we have a dataset f opened from multiple files, f.time should have the concatenation.
    # The logic below seems to reconstruct time from filenames.

    if is_multi:
        tarray = []
        # If f.time matches fname length (one time per file)
        if f.time.size == len(fname):
            # Try to use existing time index if possible, but original code ignores it mostly
            # except as a base? "t + tdelta"
            # But "t" comes from f.time.to_index().
            times = f.time.values
            for t, fn in zip(times, fname):
                try:
                    hour_str = [i for i in fn.split(".") if "atmf" in i][0][-3:]
                    hour = int(hour_str)
                    tdelta = Timedelta(hour, unit="h")
                    tarray.append(pd.Timestamp(t) + tdelta)  # Assuming t is base time?
                except:
                    tarray.append(t)
            time = to_datetime(tarray)
            f["time"] = time
    else:
        # Single file
        fn = fname[0] if isinstance(fname, (list, tuple)) else fname
        try:
            hour_str = [i for i in fn.split(".") if "atmf" in i][0][-3:]
            hour = int(hour_str)
            tdelta = Timedelta(hour, unit="h")
            # f.time might be size > 1 if single file has multiple times?
            # Original: time = f.time.to_index() + tdelta
            f["time"] = f.time.to_index() + tdelta
        except:
            pass

    return f


def _fix_nemsio(f):
    f = _rename_func(f, {})
    try:
        f["geohgt"] = _calc_nemsio_hgt(f)
    except Exception:
        pass  # print("geoht calculation not completed")
    return f


def _rename_func(f, rename_dict):
    final_dict = {}
    for i in f.data_vars.keys():
        if "midlayer" in i:
            rename_dict[i] = i.split("midlayer")[0]
    for i in rename_dict.keys():
        if i in f.data_vars.keys():
            final_dict[i] = rename_dict[i]
    f = f.rename(final_dict)
    try:
        f = f.rename({"pp25": "pm25", "pp10": "pm10"})
    except ValueError:
        pass  # print("PM25 and PM10 are not available")
    return f


def _fix_grib2(f):
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
        "AOTK_chemical_Dust_Dry_aerosol_size__2e_05_aerosol_wavelength_5_45e_07_5_65e_07_entireatmosphere": "dust25aod550",
        "var0_20_112_chemical_Dust_Dry_aerosol_size__2e_05_aerosol_wavelength_5_45e_07_5_65e_07_entireatmosphere": "tc_dust25aod550",
        "AOTK_chemical_Sea_Salt_Dry_aerosol_size__2e_05_aerosol_wavelength_5_45e_07_5_65e_07_entireatmosphere": "salt25aod550",
        "var0_20_112_chemical_Sea_Salt_Dry_aerosol_size__2e_05_aerosol_wavelength_5_45e_07_5_65e_07_entireatmosphere": "tc_salt25aod550",
        "AOTK_chemical_Sulphate_Dry_aerosol_size__2e_05_aerosol_wavelength_5_45e_07_5_65e_07_entireatmosphere": "sulf25aod550",
        "var0_20_112_chemical_Sulphate_Dry_aerosol_size__7e_07_aerosol_wavelength_5_45e_07_5_65e_07_entireatmosphere": "tc_sulf25aod550",
        "AOTK_chemical_Particulate_Organic_Matter_Dry_aerosol_size__2e_05_aerosol_wavelength_5_45e_07_5_65e_07_entireatmosphere": "oc25aod550",
        "var0_20_112_chemical_Particulate_Organic_Matter_Dry_aerosol_size__7e_07_aerosol_wavelength_5_45e_07_5_65e_07_entireatmosphere": "tc_sulfaod550",
        "AOTK_chemical_Black_Carbon_Dry_aerosol_size__2e_05_aerosol_wavelength_5_45e_07_5_65e_07_entireatmosphere": "bc25aod550",
        "var0_20_112_chemical_Black_Carbon_Dry_aerosol_size__7e_07_aerosol_wavelength_5_45e_07_5_65e_07_entireatmosphere": "tc_ocaod550",
        "AOTK_chemical_Total_Aerosol_aerosol_size__2e_05_aerosol_wavelength_6_2e_07_6_7e_07_entireatmosphere": "pm25aod640",
        "AOTK_chemical_Total_Aerosol_aerosol_size__2e_05_aerosol_wavelength_8_41e_07_8_76e_07_entireatmosphere": "pm25aod860",
        "AOTK_chemical_Total_Aerosol_aerosol_size__2e_05_aerosol_wavelength_1_628e_06_1_652e_06_entireatmosphere": "pm25aod1645",
        "AOTK_chemical_Total_Aerosol_aerosol_size__2e_05_aerosol_wavelength_1_1e_05_1_12e_05_entireatmosphere": "pm25aod11500",
        "COLMD_chemical_Total_Aerosol_aerosol_size__1e_05_aerosol_wavelength_____code_table_4_91_255_entireatmosphere": "tc_pm10",
        "COLMD_chemical_Total_Aerosol_aerosol_size__2_5e_06_aerosol_wavelength_____code_table_4_91_255_entireatmosphere": "tc_pm25",
        "COLMD_chemical_Dust_Dry_aerosol_size__2_5e_06_aerosol_wavelength_____code_table_4_91_255_entireatmosphere": "tc_dust25",
        "COLMD_chemical_Sea_Salt_Dry_aerosol_size__2_5e_06_aerosol_wavelength_____code_table_4_91_255_entireatmosphere": "tc_salt25",
        "COLMD_chemical_Black_Carbon_Dry_aerosol_size__2_36e_08_aerosol_wavelength_____code_table_4_91_255_entireatmosphere": "tc_bc036",
        "COLMD_chemical_Particulate_Organic_Matter_Dry_aerosol_size__4_24e_08_aerosol_wavelength_____code_table_4_91_255_entireatmosphere": "tc_oc0428",
        "COLMD_chemical_Sulphate_Dry_aerosol_size__2_5e_06_aerosol_wavelength_____code_table_4_91_255_entireatmosphere": "tc_sulf25",
        "AOTK_aerosol_EQ_Total_Aerosol_aerosol_size_LT_2eM05_aerosol_wavelength_GE_3D38eM07_LE_3D42eM07_entireatmosphere": "pm25aod340_eq",
        "ASYSFK_aerosol_EQ_Total_Aerosol_aerosol_size_LT_2eM05_aerosol_wavelength_GE_3D38eM07_LE_3D42eM07_entireatmosphere": "AF_pm25aod340",
        "SSALBK_aerosol_EQ_Total_Aerosol_aerosol_size_LT_2eM05_aerosol_wavelength_GE_3D38eM07_LE_3D42eM07_entireatmosphere": "SSA_pm25aod340",
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

    f = _rename_func(f, rename_dict)

    # Check if 'latitude'/'longitude' exist already or need renaming from lat_0/lon_0
    # Grib2 often has lat_0, lon_0

    if "latitude" not in f.coords:
        if "lat_0" in f.coords:
            f = f.rename({"lat_0": "latitude", "lon_0": "longitude"})
        elif "lat" in f.coords:
            f = f.rename({"lat": "latitude", "lon": "longitude"})

    # Create 2D grid if 1D (Meshgrid)
    # The original code did manual meshgrid logic.
    if f.latitude.ndim == 1 and f.longitude.ndim == 1:
        from numpy import meshgrid

        # Original logic implies lat/lon were 1D arrays of unique values?
        # "f['latitude'] = range(len(f.latitude))" -> this suggests original coords were 1D
        # NOTE: XarrayDriver typically gives what xarray gives.
        # If we want to replicate exactly:
        lat_vals = f.latitude.values
        lon_vals = f.longitude.values

        # Rename dims to y, x if not present
        if "lat_0" in f.dims:
            f = f.rename({"lat_0": "y", "lon_0": "x"})
        elif "latitude" in f.dims:  # If we renamed coords above
            f = f.rename({"latitude": "y", "longitude": "x"})

        lon, lat = meshgrid(lon_vals, lat_vals)
        f["longitude"] = (("y", "x"), lon)
        f["latitude"] = (("y", "x"), lat)
        f = f.set_coords(["latitude", "longitude"])

    return f


def _calc_nemsio_hgt(f):
    sfc = f.hgtsfc
    dz = f.delz
    z = dz + sfc
    z = z.rolling(z=len(f.z), min_periods=1).sum()
    z.name = "geohgt"
    z.attrs["long_name"] = "Geopotential Height"
    z.attrs["units"] = "m"
    return z
