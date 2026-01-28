"""CAMx Reader"""

import xarray as xr
from numpy import array, concatenate
from pandas import Series, to_datetime

from monetio.grids import get_latlon_ioapi, grid_from_dataset

from .base import GriddedReader, register_reader


@register_reader("camx")
class CAMxReader(GriddedReader):
    def open_dataset(
        self,
        files,
        earth_radius=6370000,
        convert_to_ppb=True,
        drop_duplicates=False,
        **kwargs,
    ):
        """
        Reads CAMx files using pseudonetcdf.
        """

        # Set default backend kwargs for CAMx if not present
        if "engine" not in kwargs:
            kwargs["engine"] = "pseudonetcdf"
        if "backend_kwargs" not in kwargs:
            kwargs["backend_kwargs"] = {"format": "uamiv"}

        # Pass preprocess to driver
        kwargs["preprocess"] = camx_preprocess

        ds = self.driver.open(files, **kwargs)

        # Post-processing

        # get the grid information
        grid = grid_from_dataset(ds, earth_radius=earth_radius)

        # assign attributes for dataset and all DataArrays
        ds = ds.assign_attrs({"proj4_srs": grid})
        for i in ds.variables:
            ds[i] = ds[i].assign_attrs({"proj4_srs": grid})
            for j in ds[i].attrs:
                if isinstance(ds[i].attrs[j], str):
                    ds[i].attrs[j] = ds[i].attrs[j].strip()
            # Original code added 'area' attribute to variables, but setting coords is better
            # ds[i] = ds[i].assign_attrs({"area": area_def})

        # ds = ds.assign_attrs(area=area_def) # This might not be serializable to netcdf easily, but internal use is fine.

        # get the times
        ds = _get_times(ds)

        # get the lat lon
        ds = _get_latlon(ds, grid)

        # get Predefined mapping tables for observations
        ds = _predefined_mapping_tables(ds)

        # rename dimensions
        ds = ds.rename({"COL": "x", "ROW": "y", "LAY": "z"})

        return ds


# -----------------------------------------------------------------------------
# Helper functions ported from monetio/models/camx.py
# -----------------------------------------------------------------------------


def camx_preprocess(dset):
    dset = add_lazy_pm25(dset)
    dset = add_lazy_pm10(dset)
    dset = add_lazy_pm_coarse(dset)
    dset = add_lazy_noy(dset)
    dset = add_lazy_nox(dset)
    return dset


def _get_times(d):
    # Check dimensions exist before accessing
    if "TFLAG" not in d.variables:
        return d

    idims = len(d.TFLAG.dims)
    if idims == 2:
        tflag1 = Series(d["TFLAG"][:, 0]).astype(str).str.zfill(7)
        tflag2 = Series(d["TFLAG"][:, 1]).astype(str).str.zfill(6)
    else:
        tflag1 = Series(d["TFLAG"][:, 0, 0]).astype(str).str.zfill(7)
        tflag2 = Series(d["TFLAG"][:, 0, 1]).astype(str).str.zfill(6)
    date = to_datetime([i + j for i, j in zip(tflag1, tflag2)], format="%Y%j%H%M%S")
    indexdates = Series(date).drop_duplicates(keep="last").index.values
    d = d.isel(TSTEP=indexdates)
    d["TSTEP"] = date[indexdates]
    return d.rename({"TSTEP": "time"})


def _get_latlon(dset, proj4_srs):
    lon, lat = get_latlon_ioapi(dset, proj4_srs)

    dset["longitude"] = xr.DataArray(lon, dims=["ROW", "COL"])
    dset["latitude"] = xr.DataArray(lat, dims=["ROW", "COL"])
    dset = dset.assign_coords(longitude=dset.longitude, latitude=dset.latitude)
    return dset


def add_lazy_pm25(d):
    keys = Series([i for i in d.variables])
    allvars = Series(fine)
    if "PM25_TOT" in keys.values:
        d["PM25"] = d["PM25_TOT"]  # Removed .chunk() as standard open handles chunks
    else:
        index = allvars.isin(keys)
        newkeys = allvars.loc[index]
        d["PM25"] = add_multiple_lazy(d, newkeys)
        d["PM25"] = d["PM25"].assign_attrs({"name": "PM2.5", "long_name": "PM2.5"})
    return d


def can_do(index):
    if index.max():
        return True
    else:
        return False


def add_lazy_pm10(d):
    keys = Series([i for i in d.variables])
    allvars = Series(concatenate([fine, coarse]))
    if "PM_TOT" in keys.values:
        d["PM10"] = d["PM_TOT"]
    else:
        index = allvars.isin(keys)
        if can_do(index):
            newkeys = allvars.loc[index]
            d["PM10"] = add_multiple_lazy(d, newkeys)
            d["PM10"] = d["PM10"].assign_attrs(
                {"name": "PM10", "long_name": "Particulate Matter < 10 microns"}
            )
    return d


def add_lazy_pm_coarse(d):
    keys = Series([i for i in d.variables])
    allvars = Series(coarse)
    index = allvars.isin(keys)
    if can_do(index):
        newkeys = allvars.loc[index]
        d["PM_COARSE"] = add_multiple_lazy(d, newkeys)
        d["PM_COARSE"] = d["PM_COARSE"].assign_attrs(
            {"name": "PM_COARSE", "long_name": "Coarse Mode Particulate Matter"}
        )
    return d


def add_lazy_noy(d):
    keys = Series([i for i in d.variables])
    allvars = Series(noy_gas)
    index = allvars.isin(keys)
    if can_do(index):
        newkeys = allvars.loc[index]
        d["NOy"] = add_multiple_lazy(d, newkeys)
        d["NOy"] = d["NOy"].assign_attrs({"name": "NOy", "long_name": "NOy"})
    return d


def add_lazy_nox(d):
    keys = Series([i for i in d.variables])
    allvars = Series(["NO", "NOX"])
    index = allvars.isin(keys)
    if can_do(index):
        newkeys = allvars.loc[index]
        d["NOx"] = add_multiple_lazy(d, newkeys)
        d["NOx"] = d["NOx"].assign_attrs({"name": "NOx", "long_name": "NOx"})
    return d


def add_multiple_lazy(dset, variables, weights=None):
    from numpy import ones

    if weights is None:
        weights = ones(len(variables))
    variables = variables.values
    new = dset[variables[0]].copy() * weights[0]
    for i, j in zip(variables[1:], weights[1:]):
        new = new + dset[i] * j
    return new


def _predefined_mapping_tables(dset):
    to_improve = {}
    to_nadp = {}
    to_aqs = {
        "OZONE": ["O3"],
        "PM2.5": ["PM25"],
        "CO": ["CO"],
        "NOY": [
            "NO",
            "NO2",
            "NO3",
            "N2O5",
            "HONO",
            "HNO3",
            "PAN",
            "PANX",
            "PNA",
            "NTR",
            "CRON",
            "CRN2",
            "CRNO",
            "CRPX",
            "OPAN",
        ],
        "NOX": ["NO", "NO2"],
        "SO2": ["SO2"],
        "NO": ["NO"],
        "NO2": ["NO2"],
        "SO4f": ["PSO4"],
        "PM10": ["PM10"],
        "NO3f": ["PNO3"],
        "ECf": ["PEC"],
        "OCf": ["OC"],
        "ETHANE": ["ETHA"],
        "BENZENE": ["BENZENE"],
        "TOLUENE": ["TOL"],
        "ISOPRENE": ["ISOP"],
        "O-XYLENE": ["XYL"],
        "WS": ["WSPD10"],
        "TEMP": ["TEMP2"],
        "WD": ["WDIR10"],
        "NAf": ["NA"],
        "NH4f": ["PNH4"],
    }
    to_airnow = {
        "OZONE": ["O3"],
        "PM2.5": ["PM25"],
        "CO": ["CO"],
        "NOY": [
            "NO",
            "NO2",
            "NO3",
            "N2O5",
            "HONO",
            "HNO3",
            "PAN",
            "PANX",
            "PNA",
            "NTR",
            "CRON",
            "CRN2",
            "CRNO",
            "CRPX",
            "OPAN",
        ],
        "NOX": ["NO", "NO2"],
        "SO2": ["SO2"],
        "NO": ["NO"],
        "NO2": ["NO2"],
        "SO4f": ["PSO4"],
        "PM10": ["PM10"],
        "NO3f": ["PNO3"],
        "ECf": ["PEC"],
        "OCf": ["OC"],
        "ETHANE": ["ETHA"],
        "BENZENE": ["BENZENE"],
        "TOLUENE": ["TOL"],
        "ISOPRENE": ["ISOP"],
        "O-XYLENE": ["XYL"],
        "WS": ["WSPD10"],
        "TEMP": ["TEMP2"],
        "WD": ["WDIR10"],
        "NAf": ["NA"],
        "NH4f": ["PNH4"],
    }
    to_crn = {}
    to_aeronet = {}
    to_cems = {}
    mapping_tables = {
        "improve": to_improve,
        "aqs": to_aqs,
        "airnow": to_airnow,
        "crn": to_crn,
        "cems": to_cems,
        "nadp": to_nadp,
        "aeronet": to_aeronet,
    }
    dset = dset.assign_attrs({"mapping_tables": mapping_tables})
    return dset


# Arrays
coarse = array(["CPRM", "CCRS"])
fine = array(
    [
        "NA",
        "PSO4",
        "PNO3",
        "PNH4",
        "PH2O",
        "PCL",
        "PEC",
        "FPRM",
        "FCRS",
        "SOA1",
        "SOA2",
        "SOA3",
        "SOA4",
    ]
)
noy_gas = array(
    [
        "NO",
        "NO2",
        "NO3",
        "N2O5",
        "HONO",
        "HNO3",
        "PAN",
        "PANX",
        "PNA",
        "NTR",
        "CRON",
        "CRN2",
        "CRNO",
        "CRPX",
        "OPAN",
    ]
)
poc = array(["SOA1", "SOA2", "SOA3", "SOA4"])
