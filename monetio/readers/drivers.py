from collections.abc import Callable
from typing import Union

import fsspec
import pandas as pd
import xarray as xr

try:
    import dask.dataframe as dd
except ImportError:
    dd = None


def _build_s3_config(storage_options: dict) -> dict:
    """Build an S3Store config dict from fsspec-style storage_options."""
    s3_config = {}
    if storage_options.get("anon", True):
        s3_config["skip_signature"] = "true"
    if (
        "client_kwargs" in storage_options
        and "region_name" in storage_options["client_kwargs"]
    ):
        s3_config["region"] = storage_options["client_kwargs"]["region_name"]
    return s3_config


def _select_store(file_list: list[str], storage_options: dict) -> tuple:
    """Select the appropriate object store based on file protocol.

    Parameters
    ----------
    file_list : list[str]
        List of file paths (all assumed to share the same protocol).
    storage_options : dict
        fsspec-style storage options (e.g. ``{"anon": True}``).

    Returns
    -------
    tuple[ObjectStoreRegistry, list[str]]
        The configured registry and (possibly updated) file list.
    """
    from obspec_utils.registry import ObjectStoreRegistry
    from obstore.store import HTTPStore, LocalStore, S3Store

    registry = ObjectStoreRegistry()

    if file_list[0].startswith("s3://"):
        bucket = file_list[0].replace("s3://", "").split("/")[0]
        config = _build_s3_config(storage_options)
        store = S3Store(bucket, config=config)
        registry.register(f"s3://{bucket}", store)
    elif file_list[0].startswith("http://") or file_list[0].startswith("https://"):
        store = HTTPStore()
        registry.register("http://", store)
        registry.register("https://", store)
    else:
        store = LocalStore(prefix="/")
        registry.register("file:///", store)
        file_list = [f"file://{f}" if not f.startswith("file://") else f for f in file_list]

    return registry, file_list


def _open_via_icechunk(vds, icechunk_repo: str, virtualizarr_file: str | None) -> xr.Dataset:
    """Store virtual references in Icechunk and return the dataset.

    Parameters
    ----------
    vds : virtualizarr.VirtualDataset
        The virtual dataset to persist.
    icechunk_repo : str
        Path (local or remote) to the Icechunk repository.
    virtualizarr_file : str | None
        Unused for Icechunk but accepted for interface consistency.

    Returns
    -------
    xr.Dataset
        The dataset opened from the Icechunk store.
    """
    try:
        import icechunk
    except ImportError:
        raise ImportError(
            "Icechunk backend requires 'icechunk'. "
            "Install with: pip install monetio[icechunk]"
        )

    repo = icechunk.Repository.open_or_create(icechunk_repo)
    session = repo.writable_session("main")
    store = session.store

    vds.virtualize.to_icechunk(store)
    session.commit("VirtualiZarr references")

    # Re-open for reading
    session = repo.readonly_session()
    return xr.open_zarr(session.store, consolidated=False)


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
    def expand_paths(path_input: str | list[str], fs=None) -> list[str]:
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
                    pass

                files = sorted(fs.glob(path_input))
                # fs.glob usually returns paths without the protocol (e.g. 'bucket/file.nc')
                if path_input.startswith("s3://") and files and not files[0].startswith("s3://"):
                    files = [f"s3://{f}" for f in files]

                if not files:
                    raise FileNotFoundError(f"No files found matching pattern: {path_input}")
                return files
            else:
                # It is a specific single file
                if not path_input.startswith("http") and not fs.exists(path_input):
                    raise FileNotFoundError(f"File not found: {path_input}")
                return [path_input]

        raise TypeError(f"Invalid path type: {type(path_input)}. Must be str or list.")


class XarrayDriver:
    """
    The unified driver for opening gridded data (NetCDF, GRIB, HDF).
    Supports S3 via fsspec.
    """

    def open(
        self,
        files: str | list[str],
        use_dask: bool = False,
        use_cubed: bool = False,
        use_virtualizarr: bool = False,
        virtualizarr_file: str | None = None,
        virtualizarr_backend: str = "kerchunk",
        icechunk_repo: str | None = None,
        **kwargs,
    ) -> xr.Dataset:
        """
        Open gridded data backend-agnostically.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path(s), URL(s), or glob pattern.
        use_dask : bool, optional
            Whether to use Dask for lazy loading, by default False.
        use_cubed : bool, optional
            Whether to use Cubed for lazy loading, by default False.
        use_virtualizarr : bool, optional
            Whether to use VirtualiZarr to create a virtual Zarr dataset, by default False.
            Useful for large datasets to avoid xarray.open_mfdataset overhead.
        virtualizarr_file : str, optional
            Path to save/load the VirtualiZarr reference JSON file. If provided and the file
            exists, the references will be loaded from it. If the file does not exist,
            the references will be computed and saved to this path.
        virtualizarr_backend : str, optional
            Backend for VirtualiZarr references. Must be ``"kerchunk"`` (default) or
            ``"icechunk"``. When ``"icechunk"`` is selected, references are stored in
            an Icechunk repository instead of a kerchunk JSON file.
        icechunk_repo : str, optional
            Path to the Icechunk repository. Required when
            ``virtualizarr_backend="icechunk"``.
        **kwargs : dict
            Additional arguments passed to xarray open functions.

        Returns
        -------
        xr.Dataset
            The loaded dataset.
        """
        # Validate virtualizarr_backend parameter
        if virtualizarr_backend not in ("kerchunk", "icechunk"):
            raise ValueError(
                f"Invalid virtualizarr_backend '{virtualizarr_backend}'. "
                "Must be 'kerchunk' or 'icechunk'."
            )
        # Prepare kwargs for xarray
        xr_kwargs = kwargs.copy()

        if use_cubed:
            try:
                import cubed  # noqa: F401
                import cubed_xarray  # noqa: F401
            except ImportError:
                raise ImportError(
                    "The 'cubed' backend requires 'cubed' and 'cubed-xarray'. "
                    "Install with `pip install cubed cubed-xarray`."
                )
            xr_kwargs["chunked_array_type"] = "cubed"

        # Expand wildcards (supports S3 globbing now)
        file_list = FileUtility.expand_paths(files)

        # Handle 'lazy' keyword: Eager by default per Aero Protocol.
        if "lazy" in xr_kwargs:
            use_dask = xr_kwargs.pop("lazy")

        # If laziness or specific chunking is requested, ensure auto-chunking is used.
        if (use_dask or use_cubed or "chunks" in xr_kwargs) and "chunks" not in xr_kwargs:
            xr_kwargs["chunks"] = {}

        # Extract MONETIO-specific keywords
        preprocess = xr_kwargs.pop("preprocess", None)
        read_method = xr_kwargs.pop("read_method", None)

        if use_virtualizarr:
            try:
                import ujson  # noqa: F401
                import zarr  # noqa: F401
                from virtualizarr import open_virtual_mfdataset
                from virtualizarr.parsers import HDFParser
            except ImportError:
                raise ImportError(
                    "virtualizarr v2 requires 'virtualizarr', 'obstore', 'obspec_utils', 'ujson', and 'zarr'. "
                    "Install with `pip install virtualizarr obstore obspec_utils ujson zarr`."
                )

            import os

            # --- Kerchunk cache: load existing refs if available ---
            refs = None
            if (
                virtualizarr_backend == "kerchunk"
                and virtualizarr_file is not None
                and os.path.exists(virtualizarr_file)
            ):
                try:
                    with open(virtualizarr_file) as f_ref:
                        refs = ujson.load(f_ref)
                except Exception as e:
                    import warnings

                    warnings.warn(f"Failed to load virtualizarr_file {virtualizarr_file}: {e}")
                    refs = None

            if refs is None:
                storage_options = dict(xr_kwargs.get("storage_options", {}))
                registry, file_list = _select_store(file_list, storage_options)

                concat_dim = xr_kwargs.get("concat_dim", "time")
                try:
                    vds = open_virtual_mfdataset(
                        file_list,
                        registry=registry,
                        parser=HDFParser(),
                        combine="nested",
                        concat_dim=concat_dim,
                    )
                except ValueError:
                    vds = open_virtual_mfdataset(
                        file_list, registry=registry, parser=HDFParser(), combine="by_coords"
                    )

                # --- Branch on backend ---
                if virtualizarr_backend == "icechunk":
                    ds = _open_via_icechunk(vds, icechunk_repo, virtualizarr_file)
                    if preprocess:
                        ds = preprocess(ds)
                    return ds

                # Kerchunk path: export refs and optionally cache them
                refs = vds.vz.to_kerchunk()

                if virtualizarr_file is not None:
                    try:
                        with open(virtualizarr_file, "w") as f_ref:
                            ujson.dump(refs, f_ref)
                    except Exception as e:
                        import warnings

                        warnings.warn(f"Failed to save virtualizarr_file {virtualizarr_file}: {e}")

            remote_protocol = "file"
            remote_options = {}
            if file_list[0].startswith("s3://"):
                remote_protocol = "s3"
                remote_options = dict(xr_kwargs.get("storage_options", {}))
                if "anon" not in remote_options:
                    remote_options["anon"] = True
            elif file_list[0].startswith("http"):
                remote_protocol = "http"
                # file_list for fsspec mapper should not start with file:// if they are local
            elif file_list[0].startswith("file://"):
                pass

            mapper = fsspec.get_mapper(
                "reference://",
                fo=refs,
                remote_protocol=remote_protocol,
                remote_options=remote_options,
            )

            # Clean up xr_kwargs for open_dataset
            mfdataset_keys = [
                "combine",
                "concat_dim",
                "parallel",
                "compat",
                "data_vars",
                "coords",
                "ids",
                "infer_order",
                "join",
                "engine",
                "storage_options",
            ]
            for k in mfdataset_keys:
                xr_kwargs.pop(k, None)

            ds = xr.open_dataset(
                mapper,
                engine="zarr",
                backend_kwargs={"consolidated": False},
                consolidated=False,
                **xr_kwargs,
            )

            if preprocess:
                ds = preprocess(ds)

            return ds

        try:
            # Case A: Single File (Optimized)
            if len(file_list) == 1:
                filename = file_list[0]

                # Remove open_mfdataset specific arguments to prevent TypeError in xr.open_dataset
                mfdataset_keys = [
                    "combine",
                    "concat_dim",
                    "parallel",
                    "compat",
                    "data_vars",
                    "coords",
                    "ids",
                    "infer_order",
                    "join",
                ]
                for k in mfdataset_keys:
                    xr_kwargs.pop(k, None)

                if read_method:
                    ds = read_method(filename, **xr_kwargs)
                else:
                    # Logic for standard engine/remote access
                    if filename.startswith("s3://") or filename.startswith("http"):
                        fs = FileUtility.get_fs(filename)
                        file_obj = fs.open(filename)
                    else:
                        file_obj = filename

                    if "engine" not in xr_kwargs:
                        try:
                            ds = xr.open_dataset(file_obj, engine="h5netcdf", **xr_kwargs)
                        except Exception:
                            ds = xr.open_dataset(file_obj, **xr_kwargs)
                    else:
                        ds = xr.open_dataset(file_obj, **xr_kwargs)

                # Apply preprocess manually
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
                    concat_dim = xr_kwargs.get("concat_dim")
                    if concat_dim is not None:
                        # If a explicit dimension is given, we use nested combination
                        # or direct concatenation if nested combine fails.
                        try:
                            return xr.combine_nested(
                                dsets,
                                concat_dim=concat_dim,
                                data_vars="minimal",
                                coords="minimal",
                                compat="override",
                            )
                        except (ValueError, TypeError):
                            return xr.concat(
                                dsets, dim=concat_dim, coords="different", data_vars="minimal"
                            )
                    else:
                        try:
                            return xr.combine_by_coords(
                                dsets,
                                data_vars="minimal",
                                coords="minimal",
                                compat="override",
                            )
                        except (ValueError, TypeError):
                            # Fallback to concat if combine_by_coords fails.
                            return xr.concat(
                                dsets, dim="time", coords="different", data_vars="minimal"
                            )

                # Standard path: use xr.open_mfdataset
                if preprocess:
                    xr_kwargs["preprocess"] = preprocess

                # If concat_dim is provided, ensure we use nested combine to avoid xarray errors
                if "concat_dim" in xr_kwargs and "combine" not in xr_kwargs:
                    xr_kwargs["combine"] = "nested"

                if "engine" not in xr_kwargs:
                    try:
                        return xr.open_mfdataset(file_list, engine="h5netcdf", **xr_kwargs)
                    except Exception:
                        return xr.open_mfdataset(file_list, **xr_kwargs)
                else:
                    return xr.open_mfdataset(file_list, **xr_kwargs)

        except Exception as e:
            raise OSError(f"XarrayDriver failed to open files. Error: {e}") from e


class PandasDriver:
    """
    The unified driver for opening tabular/point data.
    """

    def open(
        self,
        files: str | list[str],
        read_method: str | Callable = "read_csv",
        lazy: bool = False,
        meta: pd.DataFrame | pd.Series | dict | tuple | None = None,
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
            raise OSError(f"PandasDriver failed to open files. Error: {e}") from e
