"""Multi-HARM shared library.

Importable core used by the numbered phase scripts. Kept free of side effects
at import time (no model loading, no HF downloads) so scripts can import any
module they need without paying for it.
"""

__all__ = [
    "env", "io_utils", "chat", "model", "dataset",
    "signals", "calibrate", "detect", "metrics", "figures",
]
