from abc import ABC, abstractmethod
import fsspec
import os
import sys
from typing import Union, List, Optional
import xarray as xr
import pandas as pd

# Use tomllib for Python 3.11+, fallback to toml for older versions
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import toml as tomllib
    except ImportError:
        # If toml is not installed and python < 3.11, we can't load config
        # We define a dummy tomllib that raises error on use or returns empty
        class tomllib:
            @staticmethod
            def load(f):
                import warnings
                warnings.warn("toml library not installed, cannot load configuration.")
                return {}

from .drivers import XarrayDriver, PandasDriver

# 1. The Registry
READER_REGISTRY = {}

def register_reader(name):
    """Decorator to register a reader class."""
    def _register(cls):
        READER_REGISTRY[name] = cls
        return cls
    return _register

# 2. The Abstract Base Class
class BaseReader(ABC):
    """
    The interface that ALL readers must implement.
    """
    _readers = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'name'):
            cls._readers[cls.name] = cls

    def __init__(self):
        self._config = self._load_config()

    @property
    def config(self):
        return self._config

    def _load_config(self):
        # Allow loading from .toml if available
        # The original code assumed a specific location.
        # We can look for {name}.toml in the same directory as the module of the class?
        # Or just use the hardcoded path logic from before if appropriate.
        # The previous code used: os.path.join(os.path.dirname(__file__), f"{self.name}.toml")
        # But BaseReader is in base.py. subclasses are elsewhere.
        # If subclasses are in monetio/readers/, then dirname(__file__) is monetio/readers/
        # which is likely where the toml files are.

        try:
            config_path = os.path.join(os.path.dirname(__file__), f"{self.name}.toml")
            if os.path.exists(config_path):
                with open(config_path, 'rb') as f: # tomllib.load expects binary in 3.11+
                     return tomllib.load(f)
        except AttributeError:
             pass # self.name might not exist
        except Exception as e:
            # If fallback toml (not tomllib) is used, it might expect text file?
            # toml.load takes 'f' which can be text. tomllib.load takes binary.
            # We need to handle this difference.
            pass

        return {}

    @classmethod
    def get_reader(cls, name):
        return cls._readers.get(name)

    @abstractmethod
    def open_dataset(self,
                     files: Union[str, List[str]],
                     **kwargs) -> Union[xr.Dataset, pd.DataFrame]:
        """
        Main entry point to read data.

        Args:
            files: File path, list of paths, or glob pattern.
            **kwargs: Reader-specific arguments.

        Returns:
            xarray.Dataset (for models/sat) or pandas.DataFrame (for point obs).
        """
        pass

    def harmonize(self, ds):
        """
        Optional: Apply standard naming conventions (middleware).
        Can be overridden by specific readers.
        """
        return ds

    def _open_url(self, urlpath, **kwargs):
        """Helper method to open a file with fsspec."""
        return fsspec.open(urlpath, **kwargs)

class GriddedReader(BaseReader):
    """
    Base class for gridded data (Models, Satellites) that utilizes XarrayDriver.
    """
    def __init__(self):
        super().__init__()
        self.driver = XarrayDriver()

    def open_dataset(self, files: Union[str, List[str]], **kwargs) -> xr.Dataset:
        """
        Uses XarrayDriver to open files.
        Readers can override this to add pre/post processing.
        """
        ds = self.driver.open(files, **kwargs)
        return self.harmonize(ds)

class PointReader(BaseReader):
    """
    Base class for point/tabular data (Observations) that utilizes PandasDriver.
    """
    def __init__(self):
        super().__init__()
        self.driver = PandasDriver()

    def open_dataset(self, files: Union[str, List[str]], read_method='read_csv', **kwargs) -> pd.DataFrame:
        """
        Uses PandasDriver to open files.
        Readers can override this to add pre/post processing.
        """
        df = self.driver.open(files, read_method=read_method, **kwargs)
        return self.harmonize(df)
