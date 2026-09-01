"""CAMx Reader"""

from functools import partial
from typing import Any

import xarray as xr

from monetio.grids import grid_from_dataset

from .base import (
    GriddedReader,
    _add_ioapi_latlon,
    _convert_to_ppb,
    _format_units,
    _harmonize_ioapi_dims,
    _harmonize_ioapi_vars,
    _scientific_hygiene,
    add_lazy_diagnostic,
    register_reader,
)
from .camx_specs import COARSE, DIAGNOSTICS, FINE, NOY_GAS, POC
from .sat_utils import update_history


@register_reader("camx")
class CAMxReader(GriddedReader):
    """
    Reader for Comprehensive Air Quality Model with extensions (CAMx) model output files.

    Inherits from GriddedReader to provide backend-agnostic loading, coordinate harmonization,
    and diagnostic calculations for CAMx netCDF datasets.
    """

    def open_dataset(
        self,
        files: str | list[str],
        earth_radius: float = 6370000,
        convert_to_ppb: bool = True,
        drop_duplicates: bool = False,
        use_virtualizarr: bool = False,
        virtualizarr_file: str | None = None,
        virtualizarr_parser: str | None = None,
        virtualizarr_backend: str = "kerchunk",
        icechunk_repo: str | None = None,
        use_icechunk: bool = False,
        icechunk_url: str | None = None,
        use_dask: bool = False,
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Reads and harmonizes CAMx netCDF files into an xarray Dataset.

        Parameters
        ----------
        files : str or list of str
            File path(s), URL(s), or glob pattern matching CAMx output file(s).
        earth_radius : float, default 6370000
            Earth radius in meters for map projection calculations.
        convert_to_ppb : bool, default True
            Whether to convert gas species concentrations from ppmV to ppbV.
        drop_duplicates : bool, default False
            Whether to drop duplicate time steps within each file.
        use_virtualizarr : bool, default False
            Whether to use VirtualiZarr for dataset opening.
        virtualizarr_file : str or None, default None
            Path to the VirtualiZarr metadata file.
        virtualizarr_parser : str or None, default None
            The VirtualiZarr parser engine (e.g., 'hdf5').
        virtualizarr_backend : str, default "kerchunk"
            Backend engine for VirtualiZarr serialization.
        icechunk_repo : str or None, default None
            Path to local Icechunk repository.
        use_icechunk : bool, default False
            Whether to enable Icechunk repository streaming.
        icechunk_url : str or None, default None
            URL endpoint of the Icechunk repository.
        use_dask : bool, default False
            Whether to load dataset lazily using Dask arrays.
        **kwargs : Any
            Additional keyword arguments passed to the underlying driver or preprocessing.

        Returns
        -------
        xr.Dataset
            The harmonized CAMx model output dataset.

        Examples
        --------
        >>> from monetio.readers.camx import CAMxReader
        >>> reader = CAMxReader()
        >>> ds = reader.open_dataset("camx_out.nc", convert_to_ppb=True)
        >>> print(ds)
        """
        # Set default backend kwargs for CAMx if not present
        if "engine" not in kwargs:
            kwargs["engine"] = "pseudonetcdf"
            if "backend_kwargs" not in kwargs:
                kwargs["backend_kwargs"] = {"format": "uamiv"}

        # 1. Setup preprocessing
        if "preprocess" not in kwargs:
            kwargs["preprocess"] = partial(
                camx_preprocess,
                earth_radius=earth_radius,
                convert_to_ppb=convert_to_ppb,
            )
        # 2. Open the dataset using standard xarray (via XarrayDriver)
        if "combine" not in kwargs:
            kwargs["combine"] = "nested"
        if "concat_dim" not in kwargs:
            kwargs["concat_dim"] = "time"

        ds = super().open_dataset(
            files,
            use_virtualizarr=use_virtualizarr,
            virtualizarr_file=virtualizarr_file,
            virtualizarr_parser="hdf5",
            virtualizarr_backend=virtualizarr_backend,
            icechunk_repo=icechunk_repo,
            use_icechunk=use_icechunk,
            icechunk_url=icechunk_url,
            use_dask=use_dask,
            **kwargs,
        )

        # 3. Finalize
        if drop_duplicates:
            ds = ds.drop_duplicates("time")
            ds = update_history(ds, "Dropped duplicate time steps.")

        ds = self.harmonize(ds)

        # Update history
        ds = update_history(ds, "Read CAMx data.")

        return ds

    def harmonize(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Standardize variable names, metadata attributes, and coordinate dimensions.

        Parameters
        ----------
        ds : xr.Dataset
            Input CAMx dataset.

        Returns
        -------
        xr.Dataset
            Harmonized CAMx dataset with standardized variable names and cleaned attributes.

        Examples
        --------
        >>> from monetio.readers.camx import CAMxReader
        >>> reader = CAMxReader()
        >>> ds_harmonized = reader.harmonize(ds)
        """
        # 1. Standardize variable names and drop redundant ones
        ds = _harmonize_ioapi_vars(ds)

        # 2. Clean up attributes
        ds = _scientific_hygiene(ds)

        # Update history
        ds = update_history(ds, "Harmonized CAMx dataset.")

        return ds


def camx_preprocess(
    ds: xr.Dataset,
    *,
    earth_radius: float = 6370000,
    convert_to_ppb: bool = True,
) -> xr.Dataset:
    """
    Preprocess a single CAMx dataset before merging or finalizing.

    Adds lazy diagnostic calculations, assigns grid coordinate projection,
    formats time dimensions, converts unit scales, and applies scientific hygiene.

    Parameters
    ----------
    ds : xr.Dataset
        Input CAMx dataset.
    earth_radius : float, default 6370000
        Earth radius in meters for grid projection.
    convert_to_ppb : bool, default True
        Whether to convert gas species units from ppmV to ppbV.

    Returns
    -------
    xr.Dataset
        Preprocessed CAMx dataset.

    Examples
    --------
    >>> from monetio.readers.camx import camx_preprocess
    >>> ds_preprocessed = camx_preprocess(ds, convert_to_ppb=True)
    """
    # 1. Add lazy diagnostic variables
    for name, spec in DIAGNOSTICS.items():
        ds = add_lazy_diagnostic(ds, name, spec)

    # 2. Grid and Coordinates
    grid = grid_from_dataset(ds, earth_radius=earth_radius)
    if grid:
        ds = ds.assign_attrs({"proj4_srs": grid})
        ds = _add_ioapi_latlon(ds, grid)

        # Also assign proj4_srs to all data variables for compatibility
        for var in ds.data_vars:
            ds[var].attrs["proj4_srs"] = grid

    # 3. Time
    if "TFLAG" in ds.variables:
        from .base import _get_ioapi_times

        ds = _get_ioapi_times(ds)

    # 4. Units and Formatting
    if convert_to_ppb:
        ds = _convert_to_ppb(ds)
    ds = _format_units(ds)

    # 5. Rename dimensions
    ds = _harmonize_ioapi_dims(ds)

    # 6. Predefined mapping tables (backward compatibility)
    ds = _predefined_mapping_tables(ds)

    # 7. Harmonize variables (lowercase and drop redundant)
    ds = _harmonize_ioapi_vars(ds)

    # 8. Scientific Hygiene
    ds = _scientific_hygiene(ds)

    # Update history
    ds = update_history(ds, "Preprocessed CAMx data.")

    return ds


def _predefined_mapping_tables(ds: xr.Dataset) -> xr.Dataset:
    """
    Attach predefined observational network mapping tables to dataset attributes.

    Parameters
    ----------
    ds : xr.Dataset
        Input CAMx dataset.

    Returns
    -------
    xr.Dataset
        Dataset with attached `mapping_tables` attribute.

    Examples
    --------
    >>> ds_mapped = _predefined_mapping_tables(ds)
    """
    # Ported from legacy code
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
        "NOX": ["NO", "NOX"],
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
    # Duplicate for AirNow
    to_airnow = to_aqs.copy()

    mapping_tables = {
        "improve": {},
        "aqs": to_aqs,
        "airnow": to_airnow,
        "crn": {},
        "cems": {},
        "nadp": {},
        "aeronet": {},
    }
    ds = ds.assign_attrs({"mapping_tables": mapping_tables})
    return ds


# Legacy aliases for backward compatibility
fine = FINE
coarse = COARSE
noy_gas = NOY_GAS
poc = POC
