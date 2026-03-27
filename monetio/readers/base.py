import abc
from typing import TYPE_CHECKING, Any, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from ..util import ds_to_2d, force_object_strings
from .sat_utils import update_history

if TYPE_CHECKING:
    import dask.dataframe as dd

from .drivers import PandasDriver, XarrayDriver

# 1. The Registry
READER_REGISTRY = {}


def register_reader(name):
    """Decorator to register a reader class."""

    def _register(cls):
        READER_REGISTRY[name] = cls
        return cls

    return _register


# 2. The Abstract Base Class
class BaseReader(abc.ABC):
    """
    The interface that ALL readers must implement.
    """

    @abc.abstractmethod
    def open_dataset(
        self,
        files: Optional[Union[str, List[str]]] = None,
        dates: Optional[Any] = None,
        **kwargs,
    ) -> Union[xr.Dataset, pd.DataFrame]:
        """
        Main entry point to read data.

        Args:
            files: File path, list of paths, or glob pattern.
            dates: Dates to retrieve if files are not provided.
            **kwargs: Reader-specific arguments.

        Returns:
            xarray.Dataset (for models/sat) or pandas.DataFrame (for point obs).
        """
        pass

    def _standardize_dates(self, dates: Any) -> Optional[pd.DatetimeIndex]:
        """
        Standardize dates input to a pandas DatetimeIndex.
        """
        if dates is None:
            return None
        d = pd.to_datetime(dates)
        if isinstance(d, pd.DatetimeIndex):
            return d
        return pd.DatetimeIndex(np.atleast_1d(d))

    def _prepare_files(self, files: Any, dates: Any, **kwargs: Any) -> Any:
        """
        Resolve files or retrieve data if files are not provided.
        Returns either a list of files/URLs or the retrieved dataset directly.
        """
        if files is not None:
            return files

        if dates is not None:
            dates = self._standardize_dates(dates)
            if hasattr(self, "retrieve"):
                return self.retrieve(dates=dates, **kwargs)
            elif hasattr(self, "build_urls"):
                files = self.build_urls(dates, **kwargs)
                if isinstance(files, pd.Series):
                    files = files.tolist()
                elif isinstance(files, pd.DataFrame):
                    if "name" in files.columns:
                        files = files.name.tolist()
                    elif "url" in files.columns:
                        files = files.url.tolist()
                    else:
                        files = files.iloc[:, 0].tolist()
                return files

        raise ValueError("Must provide either 'files' or 'dates'.")

    def build_urls(self, dates: Any, **kwargs: Any) -> Union[str, List[str]]:
        """
        Construct URLs for the given dates.
        Must be implemented by subclasses that support retrieval.
        """
        raise NotImplementedError(
            f"Reader {self.__class__.__name__} does not implement build_urls."
        )

    def harmonize(self, ds):
        """
        Optional: Apply standard naming conventions (middleware).
        Can be overridden by specific readers.
        """
        return ds


class GriddedReader(BaseReader):
    """
    Base class for gridded data (Models, Satellites) that utilizes XarrayDriver.
    """

    def __init__(self):
        self.driver = XarrayDriver()

    def open_dataset(
        self,
        files: Optional[Union[str, List[str]]] = None,
        dates: Optional[Any] = None,
        **kwargs,
    ) -> xr.Dataset:
        """
        Uses XarrayDriver to open files.
        Readers can override this to add pre/post processing.
        """
        res = self._prepare_files(files, dates, **kwargs)
        if isinstance(res, xr.Dataset):
            return res

        try:
            ds = self.driver.open(res, **kwargs)
        except (OSError, FileNotFoundError) as e:
            # If files were provided explicitly, re-raise
            if files is not None:
                raise
            # If dates were used, it's a retrieval failure
            raise OSError(f"Failed to retrieve or open files for dates {dates}. Error: {e}")

        return self.harmonize(ds)


class PointReader(BaseReader):
    """
    Base class for point/tabular data (Observations) that utilizes PandasDriver.
    """

    fixed_location = True

    def __init__(self):
        self.driver = PandasDriver()

    def open_dataset(
        self,
        files: Optional[Union[str, List[str]]] = None,
        dates: Optional[Any] = None,
        read_method: Union[str, callable] = "read_csv",
        as_xarray: bool = True,
        lazy: bool = False,
        meta: Union[pd.DataFrame, pd.Series, dict, tuple, None] = None,
        **kwargs,
    ) -> Union[pd.DataFrame, xr.Dataset, "dd.DataFrame"]:
        """
        Retrieve and load point data.

        Parameters
        ----------
        files : Union[str, List[str]], optional
            File path, list of paths, or glob pattern.
        dates : Any, optional
            Dates to retrieve if files are not provided.
        read_method : str or callable, optional
            The pandas/dask reading method to use, by default "read_csv".
        as_xarray : bool, optional
            If True, return an xarray.Dataset, by default True.
        lazy : bool, optional
            If True, return a dask-backed object, by default False.
        meta : pd.DataFrame, pd.Series, dict, or tuple, optional
            Dask metadata to use for lazy loading, by default None.
        **kwargs : dict
            Additional arguments passed to the reader and driver.

        Returns
        -------
        Union[pd.DataFrame, xr.Dataset, dd.DataFrame]
            The loaded dataset.
        """
        res = self._prepare_files(files, dates, **kwargs)

        # Check for retrieved objects (DataFrames or Datasets)
        # Note: Dask DataFrames also match here if they are imported.
        try:
            import dask.dataframe as dd

            is_df = isinstance(res, (pd.DataFrame, dd.DataFrame))
        except ImportError:
            is_df = isinstance(res, pd.DataFrame)

        if is_df or isinstance(res, xr.Dataset):
            return res

        df = self.driver.open(res, read_method=read_method, lazy=lazy, meta=meta, **kwargs)

        df = self.harmonize(df)

        # Consistently force object strings to avoid nullable string issues in Pandas/Dask
        df = force_object_strings(df)

        if as_xarray:
            return self.to_xarray(df, **kwargs)

        return df

    def harmonize(
        self, df: Union[pd.DataFrame, "dd.DataFrame"]
    ) -> Union[pd.DataFrame, "dd.DataFrame"]:
        """
        Harmonize the dataset (standard naming, dropping NaNs).

        Parameters
        ----------
        df : Union[pd.DataFrame, "dd.DataFrame"]
            Input dataframe.

        Returns
        -------
        Union[pd.DataFrame, "dd.DataFrame"]
            Harmonized dataframe.
        """
        if "latitude" in df.columns and "longitude" in df.columns:
            df = df.dropna(subset=["latitude", "longitude"])

        # Update history if attributes exist (backend-agnostic)
        df = update_history(df, "Harmonized and dropped NaN locations.")

        return super().harmonize(df)

    def to_xarray(
        self, df: Union[pd.DataFrame, "dd.DataFrame"], expand2d: bool = True, **kwargs
    ) -> xr.Dataset:
        """
        Convert the DataFrame to an xarray Dataset in UGRID convention.
        By default, returns a 2D dataset (time, node) if expand2d=True.

        Parameters
        ----------
        df : Union[pd.DataFrame, dd.DataFrame]
            Input dataframe.
        expand2d : bool, optional
            Whether to expand to 2D (time, node) structure, by default True.
        **kwargs : dict
            Additional arguments passed to ds_to_2d (e.g. pivot).

        Returns
        -------
        xr.Dataset
            The dataset in UGRID convention.
        """
        # 1. Identify backend
        try:
            import dask.dataframe as dd

            is_dask = isinstance(df, dd.DataFrame)
        except ImportError:
            is_dask = False

        # 2. Prepare DataFrame (ensure time and siteid are columns)
        if is_dask:
            temp_df = df
        else:
            temp_df = df.copy()

        for name in ["time", "siteid"]:
            try:
                names = temp_df.index.names
            except AttributeError:
                names = [temp_df.index.name]

            if name in names:
                temp_df = temp_df.reset_index()

        # 3. Handle Backends
        # Consistently force object strings for both backends to avoid nullable string issues.
        temp_df = force_object_strings(temp_df)

        if is_dask:
            # 3a. Lazy Path
            ds = xr.Dataset()
            # Exception to "No Hidden Computes": lengths=True is required by Xarray
            # to determine dimension sizes for the Dataset structure.
            for col in temp_df.columns:
                ds[col] = (("node",), temp_df[col].to_dask_array(lengths=True))
        else:
            # 3b. Eager Path
            # Consistently use 1D for both Eager and Lazy by default.
            ds = temp_df.reset_index(drop=True).to_xarray()
            if "index" in ds.dims:
                ds = ds.rename({"index": "node"})

        # Set standard coordinates
        coords = [
            c for c in ["time", "siteid", "latitude", "longitude", "elevation"] if c in ds.data_vars
        ]
        ds = ds.set_coords(coords)

        # Ensure node coordinate is a simple integer range for both
        if "node" in ds.dims:
            ds.coords["node"] = (("node",), np.arange(ds.sizes["node"]))

        # 4. Standard Path (Consistently try 2D expansion by default)
        # The user requested 2D UGRID as default.
        if expand2d:
            # We pass kwargs to allow control over pivoting (wide_fmt or pivot)
            pivot = kwargs.get("wide_fmt", kwargs.get("pivot", True))
            ds = ds_to_2d(ds, pivot=pivot, fixed_location=self.fixed_location)

        # Add UGRID metadata
        if "node" in ds.dims:
            node_coords = []
            for c in ["longitude", "latitude", "elevation"]:
                if c in ds.coords:
                    node_coords.append(c)

            if node_coords:
                ds["mesh"] = xr.DataArray(
                    data=np.int32(0),
                    attrs={
                        "cf_role": "mesh_topology",
                        "topology_dimension": 0,
                        "node_coordinates": " ".join(node_coords),
                    },
                )

            if "latitude" in ds.coords:
                ds.coords["latitude"].attrs.update(
                    {"units": "degrees_north", "standard_name": "latitude"}
                )
            if "longitude" in ds.coords:
                ds.coords["longitude"].attrs.update(
                    {"units": "degrees_east", "standard_name": "longitude"}
                )
            if "elevation" in ds.coords:
                ds.coords["elevation"].attrs.update(
                    {"units": "m", "standard_name": "height_above_mean_sea_level"}
                )

            for var in ds.data_vars:
                if "node" in ds[var].dims:
                    ds[var].attrs.update({"mesh": "mesh", "location": "node"})

        # Copy attributes from DataFrame if they exist (e.g. history)
        if hasattr(df, "attrs"):
            for k, v in df.attrs.items():
                if k not in ds.attrs:
                    ds.attrs[k] = v
                elif k == "history":
                    ds.attrs[k] = f"{v}\n{ds.attrs[k]}"

        # Add Global Attributes
        if "Conventions" not in ds.attrs:
            ds.attrs["Conventions"] = "CF-1.8 UGRID-1.0"
        elif "UGRID-1.0" not in ds.attrs["Conventions"]:
            ds.attrs["Conventions"] += " UGRID-1.0"

        # Update history
        ds = update_history(ds, "Converted to xarray Dataset with UGRID convention.")

        return ds
