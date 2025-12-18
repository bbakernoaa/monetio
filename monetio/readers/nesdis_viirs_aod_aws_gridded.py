"""NESDIS VIIRS AOD AWS Gridded Reader"""

import pandas as pd
import xarray as xr
import s3fs
from typing import Union, List, Optional
from enum import Enum
from .base import GriddedReader, register_reader

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
        "path_template": "noaa-jpss/{satellite}/VIIRS/{satellite}_VIIRS_Aerosol_Optical_Depth_Gridded_Reprocessed/{resolution}_Degrees_Daily/{year}/",
        "file_template": "viirs_eps_{sat_name}_aod_{resolution}_deg_{date}.nc",
        "resolutions": {"0.050", "0.100", "0.250"},
    },
    AveragingTime.WEEKLY: {
        "path_template": "noaa-jpss/{satellite}/VIIRS/{satellite}_VIIRS_Aerosol_Optical_Depth_Gridded_Reprocessed/0.25_Degrees_Weekly/{year}/",
        "resolutions": {"0.250"},
    },
    AveragingTime.MONTHLY: {
        "path_template": "noaa-jpss/{satellite}/VIIRS/{satellite}_VIIRS_Aerosol_Optical_Depth_Gridded_Reprocessed/0.25_Degrees_Monthly/",
        "file_template": "viirs_aod_monthly_{sat_name}_0.250_deg_{date}.nc",
        "resolutions": {"0.250"},
    }
}

@register_reader("nesdis_viirs_aod_aws_gridded")
class NESDISVIIRSAODAWSGriddedReader(GriddedReader):
    """
    Reader for NESDIS VIIRS AOD AWS Gridded data.
    """

    def __init__(self):
        super().__init__()
        self.fs = s3fs.S3FileSystem(anon=True)

    def _validate_inputs(self, satellite: str, data_resolution: str, averaging_time: str) -> None:
        """
        Validate input parameters.
        """
        if satellite not in {s.value for s in Satellite}:
            raise ValueError(f"Invalid satellite: {satellite}. Must be one of {list(Satellite)}")

        if averaging_time not in {t.value for t in AveragingTime}:
            raise ValueError(f"Invalid averaging_time: {averaging_time}. Must be one of {list(AveragingTime)}")

        if data_resolution not in PRODUCT_CONFIG[averaging_time]["resolutions"]:
            raise ValueError(
                f"Invalid resolution {data_resolution} for {averaging_time} data. "
                f"Valid resolutions: {PRODUCT_CONFIG[averaging_time]['resolutions']}"
            )

    def _get_satellite_name(self, satellite: str) -> str:
        """Get the lowercase satellite name used in file paths."""
        return "npp" if satellite == "SNPP" else "noaa20"

    def _create_daily_aod_list(self, data_resolution: str, satellite: str, date_generated: List[pd.Timestamp]) -> List[str]:
        """
        Creates a list of daily AOD files.
        """
        self._validate_inputs(satellite, data_resolution, AveragingTime.DAILY)

        file_list = []
        sat_name = self._get_satellite_name(satellite)
        config = PRODUCT_CONFIG[AveragingTime.DAILY]

        for date in date_generated:
            file_date = date.strftime("%Y%m%d")
            year = file_date[:4]

            file_name = config["file_template"].format(
                sat_name=sat_name,
                resolution=data_resolution,
                date=file_date
            )

            prod_path = config["path_template"].format(
                satellite=satellite,
                resolution=data_resolution[:4],
                year=year
            )

            full_path = prod_path + file_name

            if self.fs.exists(full_path):
                file_list.append(full_path)
            else:
                if error_missing:
                    raise ValueError(f"File does not exist: {full_path}")
                else:
                    print(f"File does not exist: {full_path}")

        return file_list

    def _create_monthly_aod_list(self, satellite: str, date_generated: List[pd.Timestamp]) -> List[str]:
        """
        Creates a list of monthly AOD files.
        """
        self._validate_inputs(satellite, "0.250", AveragingTime.MONTHLY)

        file_list = []
        processed_months = set()
        sat_name = self._get_satellite_name(satellite)
        config = PRODUCT_CONFIG[AveragingTime.MONTHLY]

        for date in date_generated:
            year_month = date.strftime("%Y%m")
            if year_month in processed_months:
                continue

            processed_months.add(year_month)
            file_name = config["file_template"].format(
                sat_name=sat_name,
                date=year_month
            )

            prod_path = config["path_template"].format(satellite=satellite)
            full_path = prod_path + file_name

            if self.fs.exists(full_path):
                file_list.append(full_path)
            else:
                if error_missing:
                    raise ValueError(f"File does not exist: {full_path}")
                else:
                    print(f"File does not exist: {full_path}")

        return file_list

    def _create_weekly_aod_list(self, satellite: str, date_generated: List[pd.Timestamp]) -> List[str]:
        """
        Creates a list of weekly AOD files.
        """
        self._validate_inputs(satellite, "0.250", AveragingTime.WEEKLY)

        file_list = []
        config = PRODUCT_CONFIG[AveragingTime.WEEKLY]

        for date in date_generated:
            file_date = date.strftime("%Y%m%d")
            year = file_date[:4]

            prod_path = config["path_template"].format(
                satellite=satellite,
                year=year
            )

            try:
                all_files = self.fs.ls(prod_path)
                for file in all_files:
                    file_name = file.split("/")[-1]
                    date_range = file_name.split("_")[7].split(".")[0]
                    file_start, file_end = date_range.split("-")

                    if file_start <= file_date <= file_end and file not in file_list:
                        file_list.append(file)
            except Exception as e:
                if error_missing:
                    raise ValueError(str(e))
                else:
                    print(str(e))

        return file_list

    def open_dataset(self,
                     files: Union[str, List[str], None] = None,
                     date: Union[str, pd.Timestamp, pd.DatetimeIndex, None] = None,
                     satellite: str = "SNPP",
                     data_resolution: Union[float, str] = 0.1,
                     averaging_time: str = "daily",
                     **kwargs) -> xr.Dataset:
        """
        Reads NESDIS VIIRS AOD AWS Gridded data.

        Args:
            files: (Ignored for this reader, uses 'date' instead)
            date: Date(s) to download/read data for.
            satellite: 'SNPP' or 'NOAA20'.
            data_resolution: 0.05, 0.1, or 0.25.
            averaging_time: 'daily', 'weekly', or 'monthly'.
            **kwargs: Additional arguments passed to xarray.

        Returns:
            xarray.Dataset
        """

        if date is None:
             if files is not None:
                 if isinstance(files, str):
                     date = files
                 else:
                     raise ValueError("Date is required for NESDIS VIIRS AOD AWS Gridded reader.")
             else:
                raise ValueError("Date is required for NESDIS VIIRS AOD AWS Gridded reader.")

        if isinstance(date, (list, pd.DatetimeIndex)) or (isinstance(date, str) and "," in date):
             return self._open_mfdataset(
                dates=date,
                satellite=satellite,
                data_resolution=data_resolution,
                averaging_time=averaging_time
            )
        else:
            return self._open_dataset(
                date=date,
                satellite=satellite,
                data_resolution=data_resolution,
                averaging_time=averaging_time
            )

    def _open_dataset(self, date: Union[str, pd.Timestamp], satellite: str, data_resolution: Union[float, str], averaging_time: str) -> xr.Dataset:
        """Open single dataset."""
        self._validate_inputs(satellite, str(data_resolution).ljust(5, "0"), averaging_time)

        if isinstance(date, str):
            date_generated = [pd.Timestamp(date)]
        else:
            date_generated = [date]

        # Get file list based on averaging time
        if averaging_time == AveragingTime.MONTHLY:
            file_list = self._create_monthly_aod_list(satellite, date_generated)
        elif averaging_time == AveragingTime.WEEKLY:
            file_list = self._create_weekly_aod_list(satellite, date_generated)
        else:  # daily
            data_resolution = str(data_resolution).ljust(5, "0")
            file_list = self._create_daily_aod_list(data_resolution, satellite, date_generated)

        if len(file_list) == 0:
            raise ValueError(f"Files not available for {averaging_time} data and date: {date_generated[0]}")

        # Open and process dataset
        dset = xr.open_dataset(self.fs.open(file_list[0]))
        dset = dset.expand_dims(time=date_generated).set_coords(["time"])

        return dset

    def _open_mfdataset(self, dates: Union[pd.DatetimeIndex, pd.Timestamp, str], satellite: str, data_resolution: Union[float, str], averaging_time: str) -> xr.Dataset:
        """Open multiple datasets."""
        # Convert dates to DatetimeIndex
        if isinstance(dates, (str, pd.Timestamp)):
            dates = pd.DatetimeIndex([dates])
        elif not isinstance(dates, pd.DatetimeIndex):
            dates = pd.DatetimeIndex(dates)

        self._validate_inputs(satellite, str(data_resolution).ljust(5, "0"), averaging_time)

        # Get file list based on averaging time
        if averaging_time == AveragingTime.MONTHLY:
            file_list = self._create_monthly_aod_list(satellite, dates)
        elif averaging_time == AveragingTime.WEEKLY:
            file_list = self._create_weekly_aod_list(satellite, dates)
        else:  # daily
            data_resolution = str(data_resolution).ljust(5, "0")
            file_list = self._create_daily_aod_list(data_resolution, satellite, dates)

        if len(file_list) == 0:
            raise ValueError(f"Files not available for {averaging_time} data and dates: {dates}")

        if not len(file_list) == len(dates):
            raise ValueError(
                "'dates' and discovered file list are not the same length. "
                f"Check your dates input for {averaging_time} frequency."
            )

        # Process valid files and dates
        dates_good = []
        aws_files = []
        for d, f in zip(dates, file_list):
            if f is not None:
                aws_files.append(self.fs.open(f))
                dates_good.append(d)

        # Combine datasets
        dset = xr.open_mfdataset(aws_files, concat_dim="time", combine="nested")
        dset["time"] = dates_good

        return dset
