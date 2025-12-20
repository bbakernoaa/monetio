"""NESDIS VIIRS AOD AWS Gridded Reader"""
from enum import Enum
from typing import List, Union

import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader
from .drivers import XarrayDriver


class AveragingTime(str, Enum):
    """Enumeration of valid averaging time periods."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class Satellite(str, Enum):
    """Enumeration of valid satellites."""

    SNPP = "SNPP"
    NOAA20 = "NOAA20"


# Configuration dictionary for data products
PRODUCT_CONFIG = {
    AveragingTime.DAILY: {
        "path_template": "s3://noaa-jpss/{satellite}/VIIRS/{satellite}_VIIRS_Aerosol_Optical_Depth_Gridded_Reprocessed/{resolution}_Degrees_Daily/{year}/",
        "file_template": "viirs_eps_{sat_name}_aod_{resolution}_deg_{date}.nc",
        "resolutions": {"0.050", "0.100", "0.250"},
    },
    AveragingTime.WEEKLY: {
        "path_template": "s3://noaa-jpss/{satellite}/VIIRS/{satellite}_VIIRS_Aerosol_Optical_Depth_Gridded_Reprocessed/0.25_Degrees_Weekly/{year}/",
        "file_template": "viirs_eps_{sat_name}_aod_0.250_deg_{date_range}.nc",
        "resolutions": {"0.250"},
    },
    AveragingTime.MONTHLY: {
        "path_template": "s3://noaa-jpss/{satellite}/VIIRS/{satellite}_VIIRS_Aerosol_Optical_Depth_Gridded_Reprocessed/0.25_Degrees_Monthly/",
        "file_template": "viirs_aod_monthly_{sat_name}_0.250_deg_{date}.nc",
        "resolutions": {"0.250"},
    },
}


@register_reader("nesdis_viirs_aod_aws_gridded")
class NESDISVIIRSAODAWSGriddedReader(GriddedReader):
    """
    Reader for NESDIS VIIRS AOD AWS Gridded data.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.driver = XarrayDriver()

    def _validate_inputs(self, satellite: str, data_resolution: str, averaging_time: str) -> None:
        """
        Validate input parameters.
        """
        if satellite not in {s.value for s in Satellite}:
            raise ValueError(f"Invalid satellite: {satellite}. Must be one of {list(Satellite)}")

        if averaging_time not in {t.value for t in AveragingTime}:
            raise ValueError(
                f"Invalid averaging_time: {averaging_time}. Must be one of {list(AveragingTime)}"
            )

        if data_resolution not in PRODUCT_CONFIG[averaging_time]["resolutions"]:
            raise ValueError(
                f"Invalid resolution {data_resolution} for {averaging_time} data. "
                f"Valid resolutions: {PRODUCT_CONFIG[averaging_time]['resolutions']}"
            )

    def _get_satellite_name(self, satellite: str) -> str:
        """Get the lowercase satellite name used in file paths."""
        return "npp" if satellite == "SNPP" else "noaa20"

    def _generate_file_list(
        self,
        dates: pd.DatetimeIndex,
        satellite: str,
        data_resolution: str,
        averaging_time: str,
        error_missing: bool,
    ) -> List[str]:
        """Generate list of files to open."""
        file_list = []
        sat_name = self._get_satellite_name(satellite)
        config = PRODUCT_CONFIG[averaging_time]

        if averaging_time == AveragingTime.DAILY:
            for date in dates:
                file_date = date.strftime("%Y%m%d")
                year = file_date[:4]
                prod_path = config["path_template"].format(
                    satellite=satellite, resolution=data_resolution[:4], year=year
                )
                file_name = config["file_template"].format(
                    sat_name=sat_name, resolution=data_resolution, date=file_date
                )
                file_list.append(prod_path + file_name)
        elif averaging_time == AveragingTime.MONTHLY:
            for date in dates.to_period("M").unique():
                year_month = date.strftime("%Y%m")
                prod_path = config["path_template"].format(satellite=satellite)
                file_name = config["file_template"].format(sat_name=sat_name, date=year_month)
                file_list.append(prod_path + file_name)
        elif averaging_time == AveragingTime.WEEKLY:
            for date in dates:
                year = date.strftime("%Y")
                prod_path = config["path_template"].format(satellite=satellite, year=year)
                # Since we don't know the exact weekly file name, we use a wildcard
                file_list.append(f"{prod_path}viirs_eps_{sat_name}_aod_0.250_deg_*.nc")

        return file_list

    def open_dataset(
        self,
        files: Union[str, List[str], None] = None,
        date: Union[str, pd.Timestamp, pd.DatetimeIndex, None] = None,
        satellite: str = "SNPP",
        data_resolution: Union[float, str] = 0.1,
        averaging_time: str = "daily",
        error_missing: bool = False,
        **kwargs,
    ) -> xr.Dataset:
        if date is None:
            raise ValueError("Date is required for NESDIS VIIRS AOD AWS Gridded reader.")

        if isinstance(date, (str, pd.Timestamp)):
            dates = pd.DatetimeIndex([date])
        elif not isinstance(date, pd.DatetimeIndex):
            dates = pd.DatetimeIndex(date)
        else:
            dates = date

        data_res_str = f"{float(data_resolution):.3f}"
        self._validate_inputs(satellite, data_res_str, averaging_time)

        file_list = self._generate_file_list(
            dates, satellite, data_res_str, averaging_time, error_missing
        )

        try:
            if len(file_list) > 1:
                return self.driver.open(
                    file_list, concat_dim="time", combine="nested", **kwargs
                ).assign_coords(time=dates)
            else:
                return self.driver.open(file_list[0], **kwargs).expand_dims(time=dates)
        except Exception as e:
            if error_missing:
                raise
            else:
                import warnings

                warnings.warn(f"File does not exist on AWS: {e}")
                return xr.Dataset()