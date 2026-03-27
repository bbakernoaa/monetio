from typing import List, Union

import fsspec
import numpy as np
import pandas as pd
import xarray as xr

try:
    import dask.dataframe as dd
except ImportError:
    dd = None


class FileUtility:
    """
    Helper class to manage file path expansion (Local + S3 + HTTP).
    """

    @staticmethod
    def get_fs(path: str):
        """
        Returns the correct filesystem (local, s3, or http) based on the protocol.
        """
        if path.startswith("s3://"):
            # anon=True means public bucket. Use anon=False to use your AWS credentials.
            return fsspec.filesystem("s3", anon=True)
        elif path.startswith("http://") or path.startswith("https://"):
            return fsspec.filesystem("http")
        elif path.startswith("ftp://"):
            return fsspec.filesystem("ftp")
        return fsspec.filesystem("file")

    @staticmethod
    def expand_paths(path_input: Union[str, List[str]], fs=None) -> List[str]:
        """
        Converts a string (with wildcards), a single path, or a list of paths
        into a guaranteed list of file paths/objects.
        """
        # Convert Path objects to string
        if hasattr(path_input, "__fspath__"):
            path_input = str(path_input)

        # Convert pandas/numpy timestamps to string
        if isinstance(path_input, (pd.Timestamp, np.datetime64)):
            path_input = str(path_input)

        # Case 1: It's a list already
        if isinstance(path_input, list):
            return sorted([str(p) if hasattr(p, "__fspath__") else p for p in path_input])

        # Case 2: It's a single string (S3 or Local)
        if isinstance(path_input, str):
            # If no specific filesystem provided, guess it from the path
            if fs is None:
                fs = FileUtility.get_fs(path_input)

            # Use fsspec/s3fs to glob wildcards (works for s3://bucket/data/*.nc too!)
            if any(char in path_input for char in ["*", "?"]):
                # HTTP globbing is generally not supported by fsspec without specific implementation
                # For S3/Local it works.
                if path_input.startswith("http"):
                    # Fallback: treat as single file if glob chars present but http (unlikely to work)
                    # Or raise error.
                    # For now, assume S3/Local for globs.
                    pass

                files = sorted(fs.glob(path_input))
                # fs.glob usually returns paths without the protocol (e.g. 'bucket/file.nc')
                # We might need to prepend 's3://' again if it was stripped
                if path_input.startswith("s3://") and files and not files[0].startswith("s3://"):
                    files = [f"s3://{f}" for f in files]

                if not files:
                    raise FileNotFoundError(f"No files found matching pattern: {path_input}")
                return files
            else:
                # It is a specific single file
                # For http, exists() might involve HEAD request
                if not path_input.startswith("http") and not fs.exists(path_input):
                    raise FileNotFoundError(f"File not found: {path_input}")
                return [path_input]

        raise TypeError(f"Invalid path type: {type(path_input)}. Must be str or list.")


class XarrayDriver:
    """
    The unified driver for opening gridded data (NetCDF, GRIB, HDF).
    Supports S3 via fsspec.
    """

    def open(self, files: Union[str, List[str]], use_dask: bool = True, **kwargs) -> xr.Dataset:
        # Expand wildcards (supports S3 globbing now)
        file_list = FileUtility.expand_paths(files)

        if not file_list:
            raise FileNotFoundError("No files provided or found.")

        # Prepare kwargs for xarray
        xr_kwargs = kwargs.copy()

        # Remove infrastructure keywords
        for k in [
            "product",
            "siteid",
            "dates",
            "files",
            "param",
            "download",
            "local",
            "n_procs",
            "add_meta",
            "add_metadata",
            "backoff_factor",
            "find_paths",
            "retries",
            "sample_time_stamp",
            "verbose",
        ]:
            xr_kwargs.pop(k, None)

        # Handle 'lazy' keyword which is common in modern MONETIO readers but not xr.open_dataset
        if "lazy" in xr_kwargs:
            use_dask = xr_kwargs.pop("lazy")

        if use_dask and "chunks" not in xr_kwargs:
            xr_kwargs["chunks"] = {}

        # Extract preprocess if present
        preprocess = xr_kwargs.get("preprocess", None)

        try:
            # Case A: Single File (Optimized)
            if len(file_list) == 1:
                filename = file_list[0]

                # 'open_dataset' does not support 'preprocess', so we must remove it
                if "preprocess" in xr_kwargs:
                    del xr_kwargs["preprocess"]

                # Remove open_mfdataset specific arguments
                for k in [
                    "combine",
                    "concat_dim",
                    "parallel",
                    "compat",
                    "data_vars",
                    "coords",
                    "ids",
                    "infer_order",
                    "join",
                ]:
                    if k in xr_kwargs:
                        del xr_kwargs[k]

                # If S3 or HTTP, we open a file-like object to pass to xarray
                if filename.startswith("s3://") or filename.startswith("http"):
                    fs = FileUtility.get_fs(filename)
                    # 'open_dataset' needs a file object or a specific engine for remote
                    file_obj = fs.open(filename)
                    try:
                        ds = xr.open_dataset(file_obj, engine="h5netcdf", **xr_kwargs)
                    except Exception:
                        ds = xr.open_dataset(file_obj, **xr_kwargs)
                else:
                    try:
                        ds = xr.open_dataset(filename, engine="h5netcdf", **xr_kwargs)
                    except Exception:
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
                    return xr.open_mfdataset(file_list, engine="h5netcdf", **xr_kwargs)
                else:
                    try:
                        return xr.open_mfdataset(file_list, engine="h5netcdf", **xr_kwargs)
                    except Exception:
                        return xr.open_mfdataset(file_list, **xr_kwargs)

        except Exception as e:
            raise OSError(f"XarrayDriver failed to open files. Error: {e}")


class PandasDriver:
    """
    The unified driver for opening tabular/point data.
    """

    def open(
        self,
        files: Union[str, List[str]],
        read_method: Union[str, callable] = "read_csv",
        lazy: bool = False,
        meta: Union[pd.DataFrame, pd.Series, dict, tuple, None] = None,
        **kwargs,
    ) -> Union[pd.DataFrame, "dd.DataFrame"]:
        file_list = FileUtility.expand_paths(files)

        # Remove infrastructure keywords
        kwargs = kwargs.copy()
        for k in [
            "dates",
            "files",
            "download",
            "local",
            "n_procs",
            "as_xarray",
            "expand2d",
            "add_meta",
            "add_metadata",
            "backoff_factor",
            "find_paths",
            "retries",
            "sample_time_stamp",
            "verbose",
        ]:
            kwargs.pop(k, None)

        # Get the actual reading function
        if callable(read_method):
            reader_func = read_method
        elif hasattr(pd, read_method):
            reader_func = getattr(pd, read_method)
        else:
            raise ValueError(f"Pandas method '{read_method}' not found and not callable.")

        if lazy:
            import dask
            import dask.dataframe as dd

            # Extract preprocess if present
            preprocess = kwargs.pop("preprocess", None)

            delayed_dfs = []
            for f in file_list:
                if f.startswith("s3://"):
                    if "storage_options" not in kwargs:
                        kwargs["storage_options"] = {"anon": True}

                d = dask.delayed(reader_func)(f, **kwargs)
                if preprocess:
                    d = dask.delayed(preprocess)(d)
                delayed_dfs.append(d)

            if not delayed_dfs:
                return dd.from_pandas(pd.DataFrame(), npartitions=1)

            return dd.from_delayed(delayed_dfs, meta=meta)

        data_frames = []
        # Reuse our filesystem logic
        try:
            # Extract preprocess if present
            preprocess = kwargs.pop("preprocess", None)

            for f in file_list:
                if f.startswith("s3://"):
                    # Pandas can read S3 URLs directly if s3fs is installed!
                    # We just pass the URL string "s3://bucket/file.csv"
                    # optionally storage_options={'anon': True} can be passed in kwargs
                    if "storage_options" not in kwargs:
                        kwargs["storage_options"] = {"anon": True}  # Default to public
                    df = reader_func(f, **kwargs)
                else:
                    df = reader_func(f, **kwargs)

                if preprocess:
                    df = preprocess(df)
                data_frames.append(df)

            if not data_frames:
                return pd.DataFrame()

            return pd.concat(data_frames, ignore_index=True)

        except (RuntimeError, ValueError):
            raise
        except Exception as e:
            raise OSError(f"PandasDriver failed to open files. Error: {e}")
