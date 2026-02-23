from typing import Any, List, Optional, Union

import xarray as xr

from .core import GriddedReader
from .drivers import XarrayDriver
from .time_utils import parse_ioapi_times


class CMAQReader(GriddedReader):
    """CMAQ IOAPI File Reader following the Aero Protocol."""

    def __init__(self) -> None:
        """Initialize the CMAQ Reader."""
        super().__init__()
        self.history_message = "Optimized time parsing and modernized via Aero Protocol"

    def open_dataset(
        self,
        files: Union[str, List[str]],
        earth_radius: float = 6370000.0,
        convert_to_ppb: bool = True,
        drop_duplicates: bool = False,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Method to open CMAQ IOAPI netcdf files.

        Parameters
        ----------
        files : Union[str, List[str]]
            The path to the file or files. Supports wildcards.
        earth_radius : float, optional
            The earth radius used for the map projection, by default 6370000.0.
        convert_to_ppb : bool, optional
            If true the units of the gas species will be converted to ppbV, by default True.
        drop_duplicates : bool, optional
            If true, drops duplicate time steps, by default False.
        **kwargs : Any
            Additional arguments passed to xarray.open_dataset or xarray.open_mfdataset.

        Returns
        -------
        xr.Dataset
            The modernized CMAQ dataset.
        """
        ds = XarrayDriver.open_dataset(files, **kwargs)

        # Apply modernization steps
        ds = self._prepare_ds(ds, earth_radius, convert_to_ppb, drop_duplicates)

        return ds

    def _prepare_ds(
        self, ds: xr.Dataset, earth_radius: float, convert_to_ppb: bool, drop_duplicates: bool
    ) -> xr.Dataset:
        """Internal method to prepare the CMAQ dataset.

        Parameters
        ----------
        ds : xr.Dataset
            The raw dataset.
        earth_radius : float
            Earth radius for projection.
        convert_to_ppb : bool
            Whether to convert units to ppbV.
        drop_duplicates : bool
            Whether to drop duplicate times.

        Returns
        -------
        xr.Dataset
            The processed dataset.
        """
        # 1. Fix times using optimized vectorized parser
        ds = self._fix_times(ds, drop_duplicates)

        # 2. Rename dimensions
        rename_dict = {"COL": "x", "ROW": "y", "LAY": "z"}
        # Only rename if they exist
        rename_dict = {k: v for k, v in rename_dict.items() if k in ds.dims}
        ds = ds.rename(rename_dict)

        # 3. Get Lat/Lon and Projection
        ds = self._get_latlon(ds, earth_radius)

        # 4. Unit conversions
        if convert_to_ppb:
            ds = self._convert_units(ds)

        # 5. Add lazy diagnostic variables
        ds = self.add_lazy_diagnostics(ds)

        # 6. Update history
        ds = self.update_history(ds)

        return ds

    def _fix_times(self, ds: xr.Dataset, drop_duplicates: bool) -> xr.Dataset:
        """Vectorized time fixing.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset with TFLAG.
        drop_duplicates : bool
            Whether to drop duplicates.

        Returns
        -------
        xr.Dataset
            Dataset with proper time coordinate.
        """
        if "TFLAG" in ds:
            times = parse_ioapi_times(ds.TFLAG)
            ds = ds.assign_coords(time=times)
            if "TSTEP" in ds.dims:
                ds = ds.swap_dims({"TSTEP": "time"})
            elif "TSTEP" in ds.coords:
                ds = ds.rename({"TSTEP": "time_step"})

            if drop_duplicates:
                ds = ds.drop_duplicates("time")

        return ds

    def _get_latlon(self, ds: xr.Dataset, earth_radius: float) -> xr.Dataset:
        """Generate latitude and longitude coordinates from the grid definition.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset.
        earth_radius : float
            Radius of the earth for projection.

        Returns
        -------
        xr.Dataset
            Dataset with longitude/latitude coordinates.
        """
        from ..grids import get_ioapi_pyresample_area_def, grid_from_dataset

        grid = grid_from_dataset(ds, earth_radius=earth_radius)
        area = get_ioapi_pyresample_area_def(ds, grid)

        lon, lat = area.get_lonlats()
        # We assume dimensions are already renamed to x, y
        # Note: lon/lat shape matches (y, x)
        ds["longitude"] = xr.DataArray(lon[::-1, :], dims=["y", "x"])
        ds["latitude"] = xr.DataArray(lat[::-1, :], dims=["y", "x"])
        ds = ds.assign_coords(longitude=ds.longitude, latitude=ds.latitude)
        ds.attrs["proj4_srs"] = grid
        return ds

    def _convert_units(self, ds: xr.Dataset) -> xr.Dataset:
        """Convert gas species to ppbV and standardize mass units.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset.

        Returns
        -------
        xr.Dataset
            Dataset with converted units.
        """
        with xr.set_options(keep_attrs=True):
            for var in list(ds.data_vars):
                attrs = ds[var].attrs
                if "units" in attrs:
                    unit_str = attrs["units"]
                    if isinstance(unit_str, str):
                        unit_str = unit_str.strip()
                        if "ppmV" in unit_str:
                            ds[var] = ds[var] * 1000.0
                            ds[var].attrs["units"] = "ppbV"
                        elif "micrograms" in unit_str:
                            ds[var].attrs["units"] = r"$\mu g m^{-3}$"
        return ds

    def add_lazy_diagnostics(self, ds: xr.Dataset) -> xr.Dataset:
        """Add lazy diagnostic variables (PM2.5, NOx, etc.).

        Parameters
        ----------
        ds : xr.Dataset
            The dataset.

        Returns
        -------
        xr.Dataset
            Dataset with added diagnostic variables.
        """
        # Define species groupings
        aitken = [
            "ACLI",
            "AECI",
            "ANAI",
            "ANH4I",
            "ANO3I",
            "AOTHRI",
            "APNCOMI",
            "APOCI",
            "ASO4I",
            "AORGAI",
            "AORGPAI",
            "AORGBI",
        ]
        accumulation = [
            "AALJ",
            "AALK1J",
            "AALK2J",
            "ABNZ1J",
            "ABNZ2J",
            "ABNZ3J",
            "ACAJ",
            "ACLJ",
            "AECJ",
            "AFEJ",
            "AISO1J",
            "AISO2J",
            "AISO3J",
            "AKJ",
            "AMGJ",
            "AMNJ",
            "ANAJ",
            "ANH4J",
            "ANO3J",
            "AOLGAJ",
            "AOLGBJ",
            "AORGCJ",
            "AOTHRJ",
            "APAH1J",
            "APAH2J",
            "APAH3J",
            "APNCOMJ",
            "APOCJ",
            "ASIJ",
            "ASO4J",
            "ASQTJ",
            "ATIJ",
            "ATOL1J",
            "ATOL2J",
            "ATOL3J",
            "ATRP1J",
            "ATRP2J",
            "AXYL1J",
            "AXYL2J",
            "AXYL3J",
            "AORGAJ",
            "AORGPAJ",
            "AORGBJ",
        ]

        def sum_existing(
            ds: xr.Dataset,
            name: str,
            species: List[str],
            long_name: Optional[str] = None,
            units: Optional[str] = None,
        ) -> xr.Dataset:
            existing = [s for s in species if s in ds.data_vars]
            if existing:
                res = ds[existing[0]]
                for s in existing[1:]:
                    res = res + ds[s]
                ds[name] = res
                if long_name:
                    ds[name].attrs["long_name"] = long_name
                if units:
                    ds[name].attrs["units"] = units
            return ds

        is_ppb = "O3" in ds.data_vars and ds.O3.attrs.get("units") == "ppbV"

        ds = sum_existing(ds, "PM25", aitken + accumulation, "PM2.5", r"$\mu g m^{-3}$")
        nox_units = "ppbV" if is_ppb else "ppmV"
        ds = sum_existing(ds, "NOx", ["NO", "NO2"], "NOx", nox_units)

        return ds


def open_dataset(files: Union[str, List[str]], **kwargs: Any) -> xr.Dataset:
    """Module-level entry point for opening CMAQ datasets.

    Parameters
    ----------
    files : Union[str, List[str]]
        Path to the file or files.
    **kwargs : Any
        Passed to CMAQReader.open_dataset.

    Returns
    -------
    xr.Dataset
    """
    return CMAQReader().open_dataset(files, **kwargs)


def open_mfdataset(files: Union[str, List[str]], **kwargs: Any) -> xr.Dataset:
    """Module-level entry point for opening multiple CMAQ datasets.

    Parameters
    ----------
    files : Union[str, List[str]]
        Path to the files.
    **kwargs : Any
        Passed to CMAQReader.open_dataset.

    Returns
    -------
    xr.Dataset
    """
    return CMAQReader().open_dataset(files, **kwargs)
