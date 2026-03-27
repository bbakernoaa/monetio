"""NDACC Reader."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from .base import register_reader
from .geoms import GEOMSReader
from .sat_utils import update_history

if TYPE_CHECKING:
    pass


@register_reader("ndacc")
class NDACCReader(GEOMSReader):
    """
    NDACC (Network for the Detection of Atmospheric Composition Change) Reader.
    NDACC data follows the GEOMS (Generic Earth Observation Metadata Standard).
    """

    def open_dataset(
        self,
        files: str | list[str] | None = None,
        dates: Any | None = None,
        siteid: str | None = None,
        instrument: str | None = None,
        as_xarray: bool = True,
        **kwargs: Any,
    ) -> xr.Dataset | pd.DataFrame:
        """
        Retrieve and load NDACC data.

        Parameters
        ----------
        files : Union[str, List[str]], optional
            File path, list of paths, or glob pattern.
        dates : Union[pd.DatetimeIndex, List[datetime], datetime, str], optional
            Dates to retrieve if files are not provided.
        siteid : str, optional
            Specific NDACC site folder name (e.g. 'mauna.loa.hi').
        instrument : str, optional
            Specific instrument name (e.g. 'lidar', 'ftir', 'uvvis.doas').
        as_xarray : bool, optional
            Whether to return an xarray.Dataset, by default True.
        **kwargs : Any
            Additional arguments passed to the driver.

        Returns
        -------
        Union[xr.Dataset, pd.DataFrame]
            The processed NDACC dataset.
        """
        ds = super().open_dataset(files, dates, siteid=siteid, instrument=instrument, **kwargs)

        if ds is None or (hasattr(ds, "empty") and ds.empty):
            if as_xarray:
                return xr.Dataset()
            return pd.DataFrame()

        if not as_xarray:
            return ds.to_dataframe().reset_index()

        # Update history
        ds = update_history(ds, "Read NDACC data via Aero Protocol.")

        return ds

    def build_urls(
        self,
        dates: pd.DatetimeIndex | list[datetime.datetime] | datetime.datetime | str,
        siteid: str | None = None,
        instrument: str | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """
        Construct NDACC URLs.
        Note: NDACC data is primarily hosted on the NASA LaRC DHF server.

        Parameters
        ----------
        dates : Union[pd.DatetimeIndex, List[datetime], datetime, str]
            Dates to build URLs for.
        siteid : str, optional
            Specific site folder name (e.g. 'mauna.loa.hi').
        instrument : str, optional
            Specific instrument name (e.g. 'ftir').

        Returns
        -------
        List[str]
            List of matching NDACC URLs.
        """
        if siteid is None or instrument is None:
            import warnings

            warnings.warn(
                "NDACC retrieval requires 'siteid' (folder name) and 'instrument'. "
                "See https://www-air.larc.nasa.gov/pub/NDACC/PUBLIC/stations/ for options."
            )
            return []

        dates = pd.DatetimeIndex(np.atleast_1d(pd.to_datetime(dates)))

        # NDACC Public Archive: https://www-air.larc.nasa.gov/pub/NDACC/PUBLIC/stations/
        base_url = f"https://www-air.larc.nasa.gov/pub/NDACC/PUBLIC/stations/{siteid}/{instrument}"

        # We use fsspec to list files in the directory
        import fsspec

        try:
            fs = fsspec.filesystem("https")
            all_files = fs.ls(base_url)

            # Filter by dates in filename (usually YYYYMMDD or similar)
            # Filenames: h2o_ftir_maunaloa_20230101t120000z_001.h5
            urls = []
            for f in all_files:
                if not f.endswith((".h5", ".hdf", ".hdf5", ".nc")):
                    continue

                # Check if any requested date matches the filename
                # This is a broad match; can be refined.
                f_lower = f.lower()
                for d in dates:
                    d_str = d.strftime("%Y%m%d")
                    if d_str in f_lower:
                        urls.append(f if f.startswith("http") else f"{base_url}/{f.split('/')[-1]}")
                        break
            return urls
        except Exception as e:
            import warnings

            warnings.warn(f"Failed to list NDACC files at {base_url}: {e}")
            return []

    def harmonize(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Standardize NDACC/GEOMS variable names.

        Parameters
        ----------
        ds : xr.Dataset
            Input dataset.

        Returns
        -------
        xr.Dataset
            Harmonized dataset.
        """
        # Call GEOMS harmonization first
        ds = super().harmonize(ds)

        # NDACC-specific additions: map GEOMS names to MONETIO names
        rename_dict = {
            "o3_mixing_ratio_volume": "ozone",
            "co_mixing_ratio_volume": "carbon_monoxide",
            "no2_mixing_ratio_volume": "nitrogen_dioxide",
            "no_mixing_ratio_volume": "nitrogen_monoxide",
            "ch4_mixing_ratio_volume": "methane",
        }

        actual_rename = {}
        for k, v in rename_dict.items():
            if k in ds.variables and v not in ds.variables:
                actual_rename[k] = v

        if actual_rename:
            ds = ds.rename(actual_rename)

        return ds
