"""TOLNet Profile Reader. Deprecated wrapper — use monetio.load('tolnet', ...) instead.

Visualization helpers ``tolnet_colormap`` and ``tolnet_plot`` are **not**
deprecated and remain available here.
"""

from ..readers._deprecation import deprecated_wrapper
from ..readers.tolnet import TOLNetReader  # noqa: F401


@deprecated_wrapper(
    "monetio.profile.tolnet.open_dataset",
    'monetio.load("tolnet", files=...)',
)
def open_dataset(fname, **kwargs):
    """Open a single TOLNet HDF5 file.

    Parameters
    ----------
    fname : str
        Path to the TOLNet HDF5 file.
    **kwargs : dict
        Additional arguments forwarded to ``TOLNetReader.open_dataset``.

    Returns
    -------
    xarray.Dataset
    """
    return TOLNetReader().open_dataset(files=fname, **kwargs)


@deprecated_wrapper(
    "monetio.profile.tolnet.open_mfdataset",
    'monetio.load("tolnet", files=...)',
)
def open_mfdataset(fname, **kwargs):
    """Open multiple TOLNet HDF5 files (glob pattern supported).

    Parameters
    ----------
    fname : str
        Glob pattern or path to TOLNet HDF5 files.
    **kwargs : dict
        Additional arguments forwarded to ``TOLNetReader.open_dataset``.

    Returns
    -------
    xarray.Dataset
    """
    return TOLNetReader().open_dataset(files=fname, **kwargs)


# ---------------------------------------------------------------------------
# Visualization helpers — NOT deprecated
# ---------------------------------------------------------------------------


def tolnet_colormap():
    from matplotlib.colors import ListedColormap
    from numpy import array

    Colors = [
        array([255, 140, 255]) / 255.0,
        array([221, 111, 242]) / 255.0,
        array([187, 82, 229]) / 255.0,
        array([153, 53, 216]) / 255.0,
        array([119, 24, 203]) / 255.0,
        array([0, 0, 187]) / 255.0,
        array([0, 44, 204]) / 255.0,
        array([0, 88, 221]) / 255.0,
        array([0, 132, 238]) / 255.0,
        array([0, 175, 255]) / 255.0,
        array([0, 235, 255]) / 255.0,
        array([39, 255, 215]) / 255.0,
        array([99, 255, 155]) / 255.0,
        array([163, 255, 91]) / 255.0,
        array([211, 255, 43]) / 255.0,
        array([255, 255, 0]) / 255.0,
        array([255, 207, 0]) / 255.0,
        array([255, 159, 0]) / 255.0,
        array([255, 111, 0]) / 255.0,
        array([255, 63, 0]) / 255.0,
        array([255, 0, 0]) / 255.0,
        array([216, 0, 15]) / 255.0,
        array([178, 0, 31]) / 255.0,
        array([140, 0, 47]) / 255.0,
        array([102, 0, 63]) / 255.0,
        array([52, 52, 52]) / 255.0,
        array([96, 96, 96]) / 255.0,
        array([140, 140, 140]) / 255.0,
        array([184, 184, 184]) / 255.0,
        array([228, 228, 228]) / 255.0,
        [1.0, 1.0, 1.0],
    ]
    TNcmap = ListedColormap(Colors)
    TNcmap.set_under([1, 1, 1])
    TNcmap.set_over([0, 0, 0])
    return TNcmap


def tolnet_plot(dset, var="O3MR", units="ppbv", tolnet_cmap=True, **kwargs):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_context("notebook")
    cmap = tolnet_colormap()
    Fig, Ax = plt.subplots(figsize=(9, 6))
    dsett = dset.copy()
    dsett["z"] /= 1000.0  # put in km
    dsett[var].attrs["units"] = units
    if tolnet_cmap:
        levels = [
            0.001,
            4,
            8,
            12,
            16,
            20,
            24,
            28,
            32,
            36,
            40,
            44,
            48,
            52,
            56,
            60,
            64,
            68,
            72,
            76,
            80,
            84,
            88,
            92,
            96,
            100,
            125,
            150,
            200,
            300,
            600,
        ]
        dsett[var].plot(x="time", y="z", cmap=cmap, levels=levels, ax=Ax)
    else:
        dsett[var].plot(x="time", y="z", **kwargs)
    plt.ylabel("Altitude [km]")
    plt.xlabel("Time [UTC]")
    sns.despine()
    plt.tight_layout(pad=0)
