import warnings

from .ufs import *  # noqa: F401, F403

warnings.filterwarnings("default", category=DeprecationWarning, module=__name__)
warnings.warn("_rrfs_cmaq_mm is deprecated. Use ufs instead.", DeprecationWarning)
