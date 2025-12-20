"""NESDIS VIIRS NDVI AWS Gridded Reader"""

from typing import List, Union

import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader
from .drivers import FileUtility, XarrayDriver

# Configuration dictionary for different data products
DATA_CONFIGS = {
    "vhi": {
        "viirs": {"path": "s3://noaa-cdr-ndvi-pds/data/", "pattern": "VIIRS-Land_*"},
        "avhrr": {"path": "s3://noaa-cdr-vegetation-health-pds/data/", "pattern": "AVHRR-Land_*"},
    },
    "lai_fpar": {
        "viirs": {"path": "s3://noaa-cdr-leaf-area-index-fapar-pds/data/", "pattern": "VIIRS-Land_*"},
        "avhrr": {"path": "s3://noaa-cdr-leaf-area-index-fapar-pds/data/", "pattern": "AVHRR-Land_*"},
    },
    "snow": {
        "ims": {"path": "s3://noaa-cdr-snow-cover-extent-ims-nrt/", "pattern": "snow_cover_extent_*"}
    },
}


@register_reader("nesdis_viirs_ndvi_aws_gridded")
class NESDISVIIRSNDVIAWSGriddedReader(GriddedReader):
    """
    Reader for NESDIS VIIRS NDVI AWS Gridded data.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fs = None
        self.driver = XarrayDriver()

    def _validate_inputs(
        self, date_generated: List[pd.Timestamp], data_type: str, sensor: str
    ) -> None:
        """
        Validates input parameters.
        """
        if data_type not in DATA_CONFIGS:
            raise ValueError(
                f"Unsupported data type: {data_type}. Available types: {list(DATA_CONFIGS.keys())}"
            )

        if sensor not in DATA_CONFIGS[data_type]:
            raise ValueError(
                f"Unsupported sensor '{sensor}' for data type '{data_type}'. "
                f"Available sensors: {list(DATA_CONFIGS[data_type].keys())}"
            )

    def _get_cached_file_list(
        self, year: str, prod_path: str, pattern: str, file_date: str
    ) -> List[str]:
        """
        Cached version of file listing to improve performance for repeated requests.
        """
        path_to_glob = f"{prod_path}{year}/{pattern}{file_date}_*.nc"
        if self.fs is None:
            self.fs = FileUtility.get_fs(path_to_glob)
        return self.fs.glob(path_to_glob)

    def _create_daily_data_list(
        self,
        date_generated: List[pd.Timestamp],
        data_type: str = "vhi",
        sensor: str = "viirs",
        warning: bool = False,
    ) -> List[str]:
        """
        Creates a list of daily data files and calculates the total size of the files.
        """
        self._validate_inputs(date_generated, data_type, sensor)

        file_list = []
        config = DATA_CONFIGS[data_type][sensor]

        for date in date_generated:
            file_date = date.strftime("%Y%m%d")
            year = file_date[:4]

            try:
                file_names = self._get_cached_file_list(
                    year, config["path"], config["pattern"], file_date
                )

                if file_names:
                    file_list.extend(file_names)
                else:
                    raise FileNotFoundError(
                        f"No files found for {data_type} ({sensor}) on {file_date}"
                    )

            except Exception as e:
                if warning:
                    import warnings

                    warnings.warn(str(e))
                    file_list.append(None)
                else:
                    raise ValueError(str(e))

        return file_list

    def _process_timeofday(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Process TIMEOFDAY variable in dataset.
        """
        if "TIMEOFDAY" in dataset:
            m = dataset["TIMEOFDAY"].attrs.pop("scale_factor")
            b = dataset["TIMEOFDAY"].attrs.pop("add_offset")
            fv = dataset["TIMEOFDAY"].attrs.pop("_FillValue")

            dataset["TIMEOFDAY"] = dataset["TIMEOFDAY"] * m + b
            dataset["TIMEOFDAY"].attrs.update(units="hours")
            dataset = xr.decode_cf(dataset)

            dataset["TIMEOFDAY"] = dataset["TIMEOFDAY"].where(
                dataset["TIMEOFDAY"] != pd.Timedelta(fv * m + b, unit="hours")
            )
        else:
            dataset = xr.decode_cf(dataset)

        return dataset

    def open_dataset(
        self,
        files: Union[str, List[str], None] = None,
        date: Union[str, pd.Timestamp, pd.DatetimeIndex, None] = None,
        data_type: str = "vhi",
        sensor: str = "viirs",
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads NESDIS VIIRS NDVI AWS Gridded data.

        Args:
            files: (Ignored for this reader, uses 'date' instead)
            date: Date(s) to download/read data for.
            data_type: 'vhi', 'lai_fpar', or 'snow'.
            sensor: 'viirs', 'avhrr', or 'ims'.
            **kwargs: Additional arguments passed to xarray.

        Returns:
            xarray.Dataset
        """

        if date is None:
            if files is not None:
                if isinstance(files, str):
                    date = files
                else:
                    raise ValueError("Date is required for NESDIS VIIRS NDVI AWS Gridded reader.")
            else:
                raise ValueError("Date is required for NESDIS VIIRS NDVI AWS Gridded reader.")

        if isinstance(date, (list, pd.DatetimeIndex)) or (isinstance(date, str) and "," in date):
            return self._open_mfdataset(dates=date, data_type=data_type, sensor=sensor, **kwargs)
        else:
            return self._open_dataset(date=date, data_type=data_type, sensor=sensor, **kwargs)

    def _open_dataset(
        self,
        date: Union[str, pd.Timestamp],
        data_type: str = "vhi",
        sensor: str = "viirs",
        **kwargs,
    ) -> xr.Dataset:
        """Opens a dataset for the given date."""
        date_generated = [pd.Timestamp(date)] if isinstance(date, str) else [date]

        file_list = self._create_daily_data_list(date_generated, data_type=data_type, sensor=sensor)
        file_list = [f for f in file_list if f is not None]

        if len(file_list) == 0:
            raise ValueError(
                f"Files not available for {data_type} ({sensor}) and date: {date_generated[0]}"
            )

        dset = self.driver.open(file_list[0], decode_cf=False, **kwargs)
        return self._process_timeofday(dset)

    def _open_mfdataset(
        self,
        dates: Union[pd.DatetimeIndex, pd.Timestamp, str],
        data_type: str = "vhi",
        sensor: str = "viirs",
        error_missing: bool = False,
        **kwargs,
    ) -> xr.Dataset:
        """Opens and combines multiple NetCDF files into a single dataset."""
        if isinstance(dates, (str, pd.Timestamp)):
            dates = pd.DatetimeIndex([dates])
        elif not isinstance(dates, pd.DatetimeIndex):
            dates = pd.DatetimeIndex(dates)

        file_list = self._create_daily_data_list(
            dates, data_type=data_type, sensor=sensor, warning=not error_missing
        )

        aws_files = [f for f in file_list if f is not None]
        if len(aws_files) == 0:
            raise ValueError(f"Files not available for {data_type} ({sensor}) and dates: {dates}")

        dset = self.driver.open(aws_files, decode_cf=False, **kwargs)

        return self._process_timeofday(dset)
