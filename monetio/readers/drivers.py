import warnings
from typing import List, Union

import fsspec
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

    def open(self, files: Union[str, List[str]], use_dask: bool = False, **kwargs) -> xr.Dataset:
        # Expand wildcards (supports S3 globbing now)
        file_list = FileUtility.expand_paths(files)

        # Prepare kwargs for xarray
        xr_kwargs = kwargs.copy()

        # Handle 'lazy' keyword
        if "lazy" in xr_kwargs:
            use_dask = xr_kwargs.pop("lazy")

        # If laziness or specific chunking is requested, ensure Dask auto-chunking is used.
        if use_dask and "chunks" not in xr_kwargs:
            xr_kwargs["chunks"] = {}

        # Extract MONETIO-specific keywords
        preprocess = xr_kwargs.pop("preprocess", None)
        read_method = xr_kwargs.pop("read_method", None)

        try:
            # Case A: Single File (Optimized)
            if len(file_list) == 1:
                filename = file_list[0]

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

                if read_method:
                    ds = read_method(filename, **xr_kwargs)
                else:
                    # Logic for standard engine/remote access
                    if filename.startswith("s3://") or filename.startswith("http"):
                        fs = FileUtility.get_fs(filename)
                        file_obj = fs.open(filename)
                    else:
                        file_obj = filename

                    if "engine" in xr_kwargs:
                        ds = xr.open_dataset(file_obj, **xr_kwargs)
                    else:
                        try:
                            ds = xr.open_dataset(file_obj, engine="h5netcdf", **xr_kwargs)
                        except Exception:
                            ds = xr.open_dataset(file_obj, **xr_kwargs)

                # Apply preprocess manually for single file
                if preprocess:
                    ds = preprocess(ds)

                return ds

            # Case B: Multiple Files (dataset)
            else:
                if read_method:
                    # Custom read_method path (e.g. TOLNet)
                    dsets = [read_method(f, **xr_kwargs) for f in file_list]

                    if preprocess:
                        dsets = [preprocess(ds) for ds in dsets]

                    # Combine logic (backend-agnostic)
                    try:
                        return xr.combine_by_coords(
                            dsets,
                            data_vars="minimal",
                            coords="minimal",
                            compat="override",
                        )
                    except ValueError:
                        # Fallback to concat if combine_by_coords fails.
                        return xr.concat(
                            dsets,
                            dim=xr_kwargs.get("concat_dim", "time"),
                            coords="different",
                            data_vars="minimal",
                        )

                # Standard path: use xr.open_mfdataset
                if preprocess:
                    xr_kwargs["preprocess"] = preprocess

                if "engine" in xr_kwargs:
                    return xr.open_mfdataset(file_list, **xr_kwargs)

                # Fallback engine logic
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
