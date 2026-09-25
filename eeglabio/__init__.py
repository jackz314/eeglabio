from importlib.metadata import version as _version

try:
    __version__ = _version("eeglabio")
except Exception:
    __version__ = "0.0.0"

from . import epochs
from . import raw
from . import utils

__all__ = ['__version__', 'epochs', 'raw', 'utils']
