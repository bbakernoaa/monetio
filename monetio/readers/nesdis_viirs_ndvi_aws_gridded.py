"""NESDIS VIIRS NDVI AWS Gridded Reader"""

from typing import List, Union

import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader
from .drivers import XarrayDriver

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
        self.driver = XarrayDriver()

    def _validate_inputs(self, data_type: str, sensor: str) -> None:
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

    def _generate_file_list(self, dates: pd.DatetimeIndex, data_type: str, sensor: str) -> List[str]:
        """Generate list of files to open."""
        self._validate_inputs(data_type, sensor)
        config = DATA_CONFIGS[data_type][sensor]
        file_list = []
        for date in dates:
            year = date.strftime("%Y")
            file_date = date.strftime("%Y%m%d")
            path_to_glob = f"{config['path']}{year}/{config['pattern']}{file_date}_*.nc"
            file_list.append(path_to_glob)
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
        error_missing: bool = False,
        **kwargs,
    ) -> xr.Dataset:
        if date is None:
            raise ValueError("Date is required for NESDIS VIIRS NDVI AWS Gridded reader.")

        if isinstance(date, (str, pd.Timestamp)):
            dates = pd.DatetimeIndex([date])
        elif not isinstance(date, pd.DatetimeIndex):
            dates = pd.DatetimeIndex(date)
        else:
            dates = date

        file_list = self._generate_file_list(dates, data_type, sensor)

        try:
            if len(file_list) > 1:
                dset = self.driver.open(
                    file_list,
                    concat_dim="time",
                    combine="nested",
                    decode_cf=False,
                    **kwargs,
                )
                if not dset:
                    return xr.Dataset()
                dset = dset.assign_coords(time=dates)
            else:
                dset = self.driver.open(file_list[0], decode_cf=False, **kwargs)
                if not dset:
                    return xr.Dataset()
                dset = dset.expand_dims(time=dates)

            return self._process_timeofday(dset)
        except Exception as e:
            if error_missing:
                raise
            else:
                import warnings

                warnings.warn(f"No files found for {data_type} ({sensor}) on {dates}: {e}")
                return xr.Dataset()
