"""Utility functions for DSCIM."""

from dscim.utils.compat import (
    ensure_dataframe,
    ensure_dataarray,
    ensure_dataset,
    get_unique_values,
    safe_indexing,
    safe_reset_index,
    safe_to_dataframe,
    set_compat_mode,
    get_compat_mode,
    CompatibilityWarning,
    ConversionMode,
)

__all__ = [
    'ensure_dataframe',
    'ensure_dataarray',
    'ensure_dataset',
    'get_unique_values',
    'safe_indexing',
    'safe_reset_index',
    'safe_to_dataframe',
    'set_compat_mode',
    'get_compat_mode',
    'CompatibilityWarning',
    'ConversionMode',
]
