"""
Compatibility layer for DataFrame ↔ DataArray conversions.

This module provides transparent conversion between pandas DataFrames
and xarray DataArrays/Datasets to maintain backward compatibility
during the migration to xarray-based APIs.

Examples
--------
>>> from dscim.utils.compat import ensure_dataframe, ensure_dataarray
>>> import xarray as xr
>>>
>>> # Convert DataArray to DataFrame
>>> da = xr.DataArray([1, 2, 3], dims=['x'], coords={'x': ['a', 'b', 'c']})
>>> df = ensure_dataframe(da, name='values')
>>>
>>> # Convert DataFrame to DataArray
>>> df = pd.DataFrame({'x': ['a', 'b', 'c'], 'values': [1, 2, 3]})
>>> da = ensure_dataarray(df, index_cols=['x'], value_col='values')
"""

import warnings
import os
import pandas as pd
import xarray as xr
import numpy as np
from typing import Union, Optional, Literal
from enum import Enum

__all__ = [
    'ensure_dataframe',
    'ensure_dataarray',
    'ensure_dataset',
    'get_unique_values',
    'safe_indexing',
    'safe_reset_index',
    'CompatibilityWarning',
    'set_compat_mode',
    'get_compat_mode',
    'ConversionMode',
]


class CompatibilityWarning(UserWarning):
    """Warning for automatic DataFrame/DataArray conversions."""
    pass


class ConversionMode(Enum):
    """Conversion mode for compatibility layer."""
    AUTO = "auto"      # Convert with warnings
    STRICT = "strict"  # Raise on mismatch
    LEGACY = "legacy"  # Silent conversion


# Global configuration
_compat_mode = ConversionMode.AUTO


def set_compat_mode(mode: Union[str, ConversionMode]) -> None:
    """
    Set global compatibility mode.

    Parameters
    ----------
    mode : str or ConversionMode
        Compatibility mode: 'auto', 'strict', or 'legacy'

    Examples
    --------
    >>> set_compat_mode('legacy')  # Suppress warnings
    >>> set_compat_mode(ConversionMode.AUTO)  # Enable warnings
    """
    global _compat_mode
    if isinstance(mode, str):
        mode = ConversionMode(mode)
    _compat_mode = mode


def get_compat_mode() -> ConversionMode:
    """
    Get current compatibility mode.

    Returns
    -------
    ConversionMode
        Current mode
    """
    return _compat_mode


def _should_warn() -> bool:
    """Check if warnings should be emitted."""
    # Check environment variable first
    env_mode = os.environ.get('DSCIM_COMPAT_MODE', '').lower()
    if env_mode == 'legacy':
        return False
    return _compat_mode == ConversionMode.AUTO


def ensure_dataframe(
    obj: Union[pd.DataFrame, xr.DataArray, xr.Dataset],
    name: Optional[str] = None,
    warn: bool = True
) -> pd.DataFrame:
    """
    Ensure input is a pandas DataFrame, converting if necessary.

    Parameters
    ----------
    obj : DataFrame, DataArray, or Dataset
        Input object to convert
    name : str, optional
        Name for DataArray when converting to DataFrame.
        If not provided and obj is a DataArray with a name, uses that name.
    warn : bool, default True
        Whether to emit compatibility warning

    Returns
    -------
    pd.DataFrame
        Input converted to DataFrame

    Raises
    ------
    TypeError
        If in STRICT mode and conversion is required, or if obj type is unsupported

    Examples
    --------
    >>> da = xr.DataArray([1, 2, 3], dims=['x'], coords={'x': ['a', 'b', 'c']}, name='values')
    >>> df = ensure_dataframe(da)
    >>> print(df)
         x  values
    0    a       1
    1    b       2
    2    c       3
    """
    if isinstance(obj, pd.DataFrame):
        return obj

    if _compat_mode == ConversionMode.STRICT:
        raise TypeError(
            f"Expected pd.DataFrame, got {type(obj).__name__}. "
            "Set DSCIM_COMPAT_MODE=auto for automatic conversion."
        )

    if warn and _should_warn():
        warnings.warn(
            f"Automatically converting {type(obj).__name__} to DataFrame. "
            "This conversion may be removed in future versions. "
            "Consider updating code to use xarray APIs directly.",
            CompatibilityWarning,
            stacklevel=3
        )

    if isinstance(obj, xr.DataArray):
        # If name not provided, try to use DataArray's name
        if name is None:
            name = obj.name if obj.name else 'value'
        df = obj.to_dataframe(name=name)
    elif isinstance(obj, xr.Dataset):
        df = obj.to_dataframe()
    else:
        raise TypeError(f"Cannot convert {type(obj).__name__} to DataFrame")

    return df.reset_index()


def ensure_dataarray(
    obj: Union[pd.DataFrame, xr.DataArray, xr.Dataset],
    index_cols: Optional[list] = None,
    value_col: Optional[str] = None,
    warn: bool = True
) -> xr.DataArray:
    """
    Ensure input is an xarray DataArray, converting if necessary.

    Parameters
    ----------
    obj : DataFrame, DataArray, or Dataset
        Input object to convert
    index_cols : list of str, optional
        Columns to use as dimensions when converting DataFrame.
        If not provided, all columns except value_col are used.
    value_col : str, optional
        Column to use as values when converting DataFrame.
        If not provided and DataFrame has one non-index column, uses that.
    warn : bool, default True
        Whether to emit compatibility warning

    Returns
    -------
    xr.DataArray
        Input converted to DataArray

    Raises
    ------
    TypeError
        If in STRICT mode and conversion is required, or if obj type is unsupported
    ValueError
        If conversion parameters are ambiguous

    Examples
    --------
    >>> df = pd.DataFrame({'x': ['a', 'b', 'c'], 'values': [1, 2, 3]})
    >>> da = ensure_dataarray(df, index_cols=['x'], value_col='values')
    >>> print(da)
    <xarray.DataArray 'values' (x: 3)>
    array([1, 2, 3])
    Coordinates:
      * x        (x) object 'a' 'b' 'c'
    """
    if isinstance(obj, xr.DataArray):
        return obj

    if _compat_mode == ConversionMode.STRICT:
        raise TypeError(
            f"Expected xr.DataArray, got {type(obj).__name__}. "
            "Set DSCIM_COMPAT_MODE=auto for automatic conversion."
        )

    if warn and _should_warn():
        warnings.warn(
            f"Automatically converting {type(obj).__name__} to DataArray. "
            "This conversion may be removed in future versions.",
            CompatibilityWarning,
            stacklevel=3
        )

    if isinstance(obj, pd.DataFrame):
        # Make a copy to avoid modifying original
        df = obj.copy()

        if index_cols:
            df = df.set_index(index_cols)

        # Convert to xarray
        xr_obj = df.to_xarray()

        # Extract the value column if specified
        if value_col:
            if value_col not in xr_obj.data_vars:
                raise ValueError(f"Column '{value_col}' not found in DataFrame")
            return xr_obj[value_col]

        # If only one data variable, return it as DataArray
        if len(xr_obj.data_vars) == 1:
            return xr_obj[list(xr_obj.data_vars)[0]]

        # Multiple variables, cannot automatically convert
        raise ValueError(
            "DataFrame has multiple value columns, cannot convert to DataArray. "
            "Specify 'value_col' parameter or use ensure_dataset() instead."
        )

    elif isinstance(obj, xr.Dataset):
        # If Dataset has single variable, extract it
        if len(obj.data_vars) == 1:
            return obj[list(obj.data_vars)[0]]
        raise ValueError(
            "Dataset has multiple variables, cannot convert to DataArray. "
            "Specify which variable to extract."
        )
    else:
        raise TypeError(f"Cannot convert {type(obj).__name__} to DataArray")


def ensure_dataset(
    obj: Union[pd.DataFrame, xr.DataArray, xr.Dataset],
    index_cols: Optional[list] = None,
    warn: bool = True
) -> xr.Dataset:
    """
    Ensure input is an xarray Dataset, converting if necessary.

    Parameters
    ----------
    obj : DataFrame, DataArray, or Dataset
        Input object to convert
    index_cols : list of str, optional
        Columns to use as dimensions when converting DataFrame
    warn : bool, default True
        Whether to emit compatibility warning

    Returns
    -------
    xr.Dataset
        Input converted to Dataset

    Raises
    ------
    TypeError
        If in STRICT mode and conversion is required, or if obj type is unsupported

    Examples
    --------
    >>> df = pd.DataFrame({'x': ['a', 'b'], 'y': [1, 2], 'z': [3, 4]})
    >>> ds = ensure_dataset(df, index_cols=['x'])
    >>> print(ds)
    <xarray.Dataset>
    Dimensions:  (x: 2)
    Coordinates:
      * x        (x) object 'a' 'b'
    Data variables:
        y        (x) int64 1 2
        z        (x) int64 3 4
    """
    if isinstance(obj, xr.Dataset):
        return obj

    if _compat_mode == ConversionMode.STRICT:
        raise TypeError(
            f"Expected xr.Dataset, got {type(obj).__name__}. "
            "Set DSCIM_COMPAT_MODE=auto for automatic conversion."
        )

    if warn and _should_warn():
        warnings.warn(
            f"Automatically converting {type(obj).__name__} to Dataset. "
            "This conversion may be removed in future versions.",
            CompatibilityWarning,
            stacklevel=3
        )

    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        if index_cols:
            df = df.set_index(index_cols)
        return df.to_xarray()
    elif isinstance(obj, xr.DataArray):
        return obj.to_dataset()
    else:
        raise TypeError(f"Cannot convert {type(obj).__name__} to Dataset")


def get_unique_values(
    obj: Union[pd.DataFrame, xr.DataArray, xr.Dataset],
    column: str
) -> np.ndarray:
    """
    Get unique values from a column/coordinate, works for both DataFrame and xarray.

    Parameters
    ----------
    obj : DataFrame, DataArray, or Dataset
        Object containing the column/coordinate
    column : str
        Name of column/coordinate

    Returns
    -------
    np.ndarray
        Array of unique values

    Raises
    ------
    KeyError
        If column/coordinate not found
    TypeError
        If obj type is unsupported

    Examples
    --------
    >>> df = pd.DataFrame({'ssp': ['SSP1', 'SSP2', 'SSP1']})
    >>> get_unique_values(df, 'ssp')
    array(['SSP1', 'SSP2'], dtype=object)

    >>> ds = xr.Dataset({'temp': (['ssp'], [1, 2])}, coords={'ssp': ['SSP1', 'SSP2']})
    >>> get_unique_values(ds, 'ssp')
    array(['SSP1', 'SSP2'], dtype=object)
    """
    if isinstance(obj, pd.DataFrame):
        if column in obj.columns:
            return obj[column].unique()
        elif hasattr(obj.index, 'names') and column in obj.index.names:
            return obj.index.get_level_values(column).unique().values
        else:
            raise KeyError(f"Column '{column}' not found in DataFrame")

    elif isinstance(obj, (xr.DataArray, xr.Dataset)):
        if column in obj.coords:
            values = obj.coords[column].values
            # Return unique values (coords are usually already unique, but just in case)
            return np.unique(values)
        elif isinstance(obj, xr.Dataset) and column in obj.data_vars:
            # If it's a data variable, get unique values
            return np.unique(obj[column].values)
        else:
            raise KeyError(f"Coordinate '{column}' not found in {type(obj).__name__}")
    else:
        raise TypeError(f"Unsupported type: {type(obj).__name__}")


def safe_indexing(
    obj: Union[pd.DataFrame, xr.Dataset, xr.DataArray],
    condition: Union[pd.Series, xr.DataArray, np.ndarray]
) -> Union[pd.DataFrame, xr.Dataset, xr.DataArray]:
    """
    Apply boolean indexing that works for both DataFrame and xarray objects.

    Parameters
    ----------
    obj : DataFrame, Dataset, or DataArray
        Object to index
    condition : Series, DataArray, or ndarray
        Boolean condition

    Returns
    -------
    DataFrame, Dataset, or DataArray
        Filtered object (same type as input)

    Raises
    ------
    TypeError
        If obj type is unsupported

    Examples
    --------
    >>> df = pd.DataFrame({'ssp': ['SSP1', 'SSP2'], 'value': [1, 2]})
    >>> safe_indexing(df, df['ssp'] == 'SSP1')
         ssp  value
    0  SSP1      1
    """
    if isinstance(obj, pd.DataFrame):
        # For DataFrame, use standard boolean indexing
        return obj[condition]

    elif isinstance(obj, (xr.Dataset, xr.DataArray)):
        # For xarray, use where with drop=True
        return obj.where(condition, drop=True)
    else:
        raise TypeError(f"Unsupported type: {type(obj).__name__}")


def safe_reset_index(
    obj: Union[pd.DataFrame, xr.DataArray, xr.Dataset]
) -> Union[pd.DataFrame, xr.DataArray, xr.Dataset]:
    """
    Reset index if DataFrame, pass through if xarray object.

    Parameters
    ----------
    obj : DataFrame, DataArray, or Dataset
        Object to reset

    Returns
    -------
    Same type as input
        DataFrame with reset index, or unchanged xarray object

    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2]}, index=['x', 'y'])
    >>> safe_reset_index(df)
      index  a
    0     x  1
    1     y  2

    >>> da = xr.DataArray([1, 2], dims=['x'])
    >>> safe_reset_index(da)  # Returns unchanged
    <xarray.DataArray (x: 2)>
    array([1, 2])
    Dimensions without coordinates: x
    """
    if isinstance(obj, pd.DataFrame):
        return obj.reset_index(drop=False)
    else:
        # xarray objects don't have traditional index to reset
        return obj


def safe_to_dataframe(
    obj: Union[pd.DataFrame, xr.DataArray, xr.Dataset],
    name: Optional[str] = None
) -> pd.DataFrame:
    """
    Convert to DataFrame safely, handling both pandas and xarray objects.

    This is a convenience function that combines ensure_dataframe with
    additional logic for common use cases.

    Parameters
    ----------
    obj : DataFrame, DataArray, or Dataset
        Object to convert
    name : str, optional
        Name for DataArray values column

    Returns
    -------
    pd.DataFrame
        Object as DataFrame

    Examples
    --------
    >>> da = xr.DataArray([1, 2, 3], dims=['x'], name='values')
    >>> safe_to_dataframe(da)
         x  values
    0    0       1
    1    1       2
    2    2       3
    """
    return ensure_dataframe(obj, name=name, warn=False)
