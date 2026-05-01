import warnings
from functools import wraps


def deprecated_wrapper(legacy_name: str, load_equivalent: str, removal_version: str = "0.4.0"):
    """
    Decorator that emits a DeprecationWarning when a legacy function is called.

    Parameters
    ----------
    legacy_name : str
        Full qualified name of the deprecated function (e.g., "monetio.models.cmaq.open_dataset").
    load_equivalent : str
        The recommended monetio.load() call (e.g., 'monetio.load("cmaq", files=...)').
    removal_version : str
        Version when the function will be removed.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{legacy_name} is deprecated and will be removed in v{removal_version}. "
                f"Use {load_equivalent} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator
