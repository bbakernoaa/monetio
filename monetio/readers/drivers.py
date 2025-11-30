import xarray as xr
import pandas as pd
import glob
import os
import s3fs
import fsspec
from typing import Union, List, Any

class FileUtility:
    """
    Helper class to manage file path expansion (Local + S3).
    """
    @staticmethod
    def get_fs(path: str):
        """
        Returns the correct filesystem (local or s3) based on the protocol.
        """
        if path.startswith("s3://"):
            # anon=True means public bucket. Use anon=False to use your AWS credentials.
            return fsspec.filesystem("s3", anon=True)
        return fsspec.filesystem("file")

    @staticmethod
    def expand_paths(path_input: Union[str, List[str]], fs=None) -> List[str]:
        """
        Converts a string (with wildcards), a single path, or a list of paths
        into a guaranteed list of file paths/objects.
        """
        # Case 1: It's a list already
        if isinstance(path_input, list):
            return sorted(path_input)

        # Case 2: It's a single string (S3 or Local)
        if isinstance(path_input, str):
            # If no specific filesystem provided, guess it from the path
            if fs is None:
                fs = FileUtility.get_fs(path_input)

            # Use fsspec/s3fs to glob wildcards (works for s3://bucket/data/*.nc too!)
            if any(char in path_input for char in ['*', '?']):
                files = sorted(fs.glob(path_input))
                # fs.glob usually returns paths without the protocol (e.g. 'bucket/file.nc')
                # We might need to prepend 's3://' again if it was stripped
                if path_input.startswith("s3://") and not files[0].startswith("s3://"):
                    files = [f"s3://{f}" for f in files]

                if not files:
                    raise FileNotFoundError(f"No files found matching pattern: {path_input}")
                return files
            else:
                # It is a specific single file
                if not fs.exists(path_input):
                    raise FileNotFoundError(f"File not found: {path_input}")
                return [path_input]

        raise TypeError(f"Invalid path type: {type(path_input)}. Must be str or list.")

class XarrayDriver:
    """
    The unified driver for opening gridded data (NetCDF, GRIB, HDF).
    Supports S3 via fsspec.
    """

    def open(self,
             files: Union[str, List[str]],
             use_dask: bool = True,
             **kwargs) -> xr.Dataset:

        # Expand wildcards (supports S3 globbing now)
        file_list = FileUtility.expand_paths(files)

        # Prepare kwargs for xarray
        xr_kwargs = kwargs.copy()
        if use_dask and 'chunks' not in xr_kwargs:
            xr_kwargs['chunks'] = {}

        # Extract preprocess if present
        preprocess = xr_kwargs.get('preprocess', None)

        try:
            # Case A: Single File (Optimized)
            if len(file_list) == 1:
                filename = file_list[0]

                # 'open_dataset' does not support 'preprocess', so we must remove it
                if 'preprocess' in xr_kwargs:
                    del xr_kwargs['preprocess']

                # Remove open_mfdataset specific arguments
                for k in ['combine', 'concat_dim', 'parallel', 'compat', 'data_vars', 'coords', 'ids', 'infer_order', 'join']:
                    if k in xr_kwargs:
                        del xr_kwargs[k]

                # If S3, we open a file-like object to pass to xarray
                if filename.startswith("s3://"):
                    fs = FileUtility.get_fs(filename)
                    # 'open_dataset' needs a file object or a specific engine for remote
                    file_obj = fs.open(filename)
                    ds = xr.open_dataset(file_obj, **xr_kwargs)
                else:
                    ds = xr.open_dataset(filename, **xr_kwargs)

                # Apply preprocess manually
                if preprocess:
                    ds = preprocess(ds)

                return ds

            # Case B: Multiple Files (dataset)
            else:
                # xr.open_mfdataset handles URLs intelligently if 'parallel=True'
                # But generally, passing a list of S3 URLs works if backend supports it.
                if file_list[0].startswith("s3://"):
                    # For S3, open_mfdataset often prefers fsspec objects explicitly
                    fs = FileUtility.get_fs(file_list[0])
                    # Create list of file objects (buffers)
                    # Note: This can be slow for 1000s of files;
                    # optimization: pass s3://.../*.nc directly to open_mfdataset if engine supports it
                    return xr.open_mfdataset(file_list, engine='h5netcdf', **xr_kwargs)
                else:
                    return xr.open_mfdataset(file_list, **xr_kwargs)

        except Exception as e:
            raise IOError(f"XarrayDriver failed to open files. Error: {e}")

class PandasDriver:
    """
    The unified driver for opening tabular/point data.
    """

    def open(self,
             files: Union[str, List[str]],
             read_method: str = 'read_csv',
             **kwargs) -> pd.DataFrame:

        file_list = FileUtility.expand_paths(files)

        # Get the actual pandas function
        if not hasattr(pd, read_method):
            raise ValueError(f"Pandas method '{read_method}' not found.")
        reader_func = getattr(pd, read_method)

        data_frames = []

        # Re-use our filesystem logic
        fs = None
        if file_list and file_list[0].startswith("s3://"):
            fs = FileUtility.get_fs(file_list[0])

        try:
            for f in file_list:
                if f.startswith("s3://"):
                    # Pandas can read S3 URLs directly if s3fs is installed!
                    # We just pass the URL string "s3://bucket/file.csv"
                    # optionally storage_options={'anon': True} can be passed in kwargs
                    if 'storage_options' not in kwargs:
                         kwargs['storage_options'] = {'anon': True} # Default to public
                    df = reader_func(f, **kwargs)
                else:
                    df = reader_func(f, **kwargs)
                data_frames.append(df)

            if not data_frames:
                return pd.DataFrame()

            return pd.concat(data_frames, ignore_index=True)

        except Exception as e:
            raise IOError(f"PandasDriver failed to open files. Error: {e}")
