from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Optional, Union

import pandas as pd
import xarray as xr

from .core import PointReader
from .drivers import PandasDriver

if TYPE_CHECKING:
    import dask.dataframe as dd


def read_crn(
    files: str | Iterable[str],
    dates: Optional[Iterable[Any]] = None,
    *,
    lazy: bool = True,
    **kwargs: Any,
) -> xr.Dataset:
    """Load CRN data.

    Parameters
    ----------
    files : str or list of str
        Path(s) to the file(s) to load.
    dates : array-like, optional
        The dates corresponding to the files.
    lazy : bool, optional
        If True, load the data lazily. Default is True.
    **kwargs
        Additional arguments passed to CRNReader.open_dataset.

    Returns
    -------
    xarray.Dataset
    """
    return CRNReader().open_dataset(files, dates, lazy=lazy, **kwargs)


class CRNReader(PointReader):
    """Reader for NOAA Climate Reference Network (CRN) hourly data.

    Follows the Aero Protocol:
    - Backend Agnostic (supports Eager and Lazy)
    - UGRID-compliant output
    - Provenance Tracking
    """

    def __init__(self) -> None:
        super().__init__()
        self.driver = PandasDriver()
        self.hcols = [
            "WBANNO",
            "UTC_DATE",
            "UTC_TIME",
            "LST_DATE",
            "LST_TIME",
            "CRX_VN",
            "LONGITUDE",
            "LATITUDE",
            "T_CALC",
            "T_AVG",
            "T_MAX",
            "T_MIN",
            "P_CALC",
            "SOLARAD",
            "SOLARAD_FLAG",
            "SOLARAD_MAX",
            "SOLARAD_MAX_FLAG",
            "SOLARAD_MIN",
            "SOLARAD_MIN_FLAG",
            "SUR_TEMP_TYPE",
            "SUR_TEMP",
            "SUR_TEMP_FLAG",
            "SUR_TEMP_MAX",
            "SUR_TEMP_MAX_FLAG",
            "SUR_TEMP_MIN",
            "SUR_TEMP_MIN_FLAG",
            "RH_AVG",
            "RH_AVG_FLAG",
            "SOIL_MOISTURE_5",
            "SOIL_MOISTURE_10",
            "SOIL_MOISTURE_20",
            "SOIL_MOISTURE_50",
            "SOIL_MOISTURE_100",
            "SOIL_TEMP_5",
            "SOIL_TEMP_10",
            "SOIL_TEMP_20",
            "SOIL_TEMP_50",
            "SOIL_TEMP_100",
        ]

    def open_dataset(
        self,
        files: str | Iterable[str],
        dates: Optional[Iterable[Any]] = None,
        *,
        lazy: bool = True,
        expand2d: bool = True,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open CRN data as a UGRID-compliant xarray Dataset.

        Parameters
        ----------
        files : str or list of str
            Path(s) to the file(s) to load.
        dates : array-like, optional
            The dates corresponding to the files.
        lazy : bool, optional
            If True, use Dask to load the data lazily. Default is True.
        expand2d : bool, optional
            If True, expand the dataset to 2D (time, node). Default is True.
        **kwargs
            Additional arguments passed to the driver's open_dataset.

        Returns
        -------
        xarray.Dataset
        """
        # 1. Load data via driver
        read_kwargs = {
            "sep": r"\s+",
            "names": self.hcols,
            "na_values": [-99999, -9999.0],
        }
        read_kwargs.update(kwargs)

        df = self.driver.open_dataset(files, dates, lazy=lazy, **read_kwargs)

        # 2. Fix time
        df = self._fix_time(df, lazy=lazy)

        # 3. Rename columns to standard names
        df = df.rename(
            columns={"WBANNO": "siteid", "LATITUDE": "latitude", "LONGITUDE": "longitude"}
        )

        # 4. Convert to xarray
        ds = self.to_xarray(df, expand2d=expand2d)

        # 5. Post-process
        ds = self._post_process(ds)

        return ds

    def _fix_time(self, df: pd.DataFrame | dd.DataFrame, lazy: bool = True) -> pd.DataFrame | dd.DataFrame:
        """Manually construct 'time' column from UTC_DATE and UTC_TIME."""
        if lazy:
            import dask.dataframe as dd

            date_str = df["UTC_DATE"].astype(str)
            time_str = df["UTC_TIME"].astype(str).str.zfill(4)
            df["time"] = dd.to_datetime(date_str + time_str, format="%Y%m%d%H%M")
        else:
            date_str = df["UTC_DATE"].astype(str)
            time_str = df["UTC_TIME"].astype(str).str.zfill(4)
            df["time"] = pd.to_datetime(date_str + time_str, format="%Y%m%d%H%M")

        return df

    def _post_process(self, ds: xr.Dataset) -> xr.Dataset:
        """CRN-specific post-processing: unit conversions and metadata."""

        def celsius_to_kelvin(da: xr.DataArray) -> xr.DataArray:
            return da + 273.15

        # Vectorized unit conversions (Celsius to Kelvin)
        # We exclude FLAG and TYPE variables which are not numeric temperatures
        temp_vars = [
            v
            for v in ds.data_vars
            if ("T_" in v or "TEMP" in v) and "FLAG" not in v and "TYPE" not in v
        ]
        for v in temp_vars:
            ds[v] = xr.apply_ufunc(
                celsius_to_kelvin,
                ds[v],
                dask="parallelized",
                keep_attrs=True,
            )
            ds[v].attrs["units"] = "K"

        if "P_CALC" in ds.data_vars:
            ds["P_CALC"].attrs["units"] = "mm"

        rh_vars = [v for v in ds.data_vars if "RH_" in v]
        for v in rh_vars:
            ds[v].attrs["units"] = "%"

        sm_vars = [v for v in ds.data_vars if "SOIL_MOISTURE" in v]
        for v in sm_vars:
            ds[v].attrs["units"] = "m^3/m^3"

        ds = self.update_history(ds, "Applied CRN-specific post-processing (unit conversions)")
        return ds
