"""Tests for compatibility layer between pandas DataFrames and xarray DataArrays/Datasets."""

import pytest
import pandas as pd
import xarray as xr
import numpy as np
import os
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


class TestCompatibilityMode:
    """Test compatibility mode configuration."""

    def test_set_get_compat_mode_string(self):
        """Test setting mode with string."""
        set_compat_mode('legacy')
        assert get_compat_mode() == ConversionMode.LEGACY

        set_compat_mode('auto')
        assert get_compat_mode() == ConversionMode.AUTO

        set_compat_mode('strict')
        assert get_compat_mode() == ConversionMode.STRICT

    def test_set_get_compat_mode_enum(self):
        """Test setting mode with enum."""
        set_compat_mode(ConversionMode.LEGACY)
        assert get_compat_mode() == ConversionMode.LEGACY

        set_compat_mode(ConversionMode.AUTO)
        assert get_compat_mode() == ConversionMode.AUTO

    def test_invalid_mode_string(self):
        """Test that invalid mode string raises error."""
        with pytest.raises(ValueError):
            set_compat_mode('invalid_mode')


class TestEnsureDataFrame:
    """Test ensure_dataframe function."""

    def test_dataframe_passthrough(self):
        """Test that DataFrame passes through unchanged."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = ensure_dataframe(df)
        pd.testing.assert_frame_equal(result, df)

    def test_dataarray_to_dataframe_with_name(self):
        """Test converting DataArray to DataFrame with explicit name."""
        set_compat_mode('legacy')  # Suppress warnings for test
        da = xr.DataArray(
            [1, 2, 3],
            dims=['x'],
            coords={'x': ['a', 'b', 'c']},
            name='original_name'
        )

        df = ensure_dataframe(da, name='values')

        assert isinstance(df, pd.DataFrame)
        assert 'values' in df.columns
        assert 'x' in df.columns
        assert len(df) == 3
        set_compat_mode('auto')

    def test_dataarray_to_dataframe_auto_name(self):
        """Test converting DataArray to DataFrame using DataArray's name."""
        set_compat_mode('legacy')
        da = xr.DataArray(
            [1, 2, 3],
            dims=['x'],
            coords={'x': ['a', 'b', 'c']},
            name='my_values'
        )

        df = ensure_dataframe(da)

        assert isinstance(df, pd.DataFrame)
        assert 'my_values' in df.columns
        set_compat_mode('auto')

    def test_dataset_to_dataframe(self):
        """Test converting Dataset to DataFrame."""
        set_compat_mode('legacy')
        ds = xr.Dataset(
            {
                'temp': (['x'], [1, 2, 3]),
                'humidity': (['x'], [4, 5, 6])
            },
            coords={'x': ['a', 'b', 'c']}
        )

        df = ensure_dataframe(ds)

        assert isinstance(df, pd.DataFrame)
        assert 'temp' in df.columns
        assert 'humidity' in df.columns
        assert len(df) == 3
        set_compat_mode('auto')

    def test_dataarray_to_dataframe_with_warning(self):
        """Test that conversion emits warning in auto mode."""
        set_compat_mode('auto')
        da = xr.DataArray([1, 2, 3], dims=['x'])

        with pytest.warns(CompatibilityWarning, match="Automatically converting"):
            df = ensure_dataframe(da)

        assert isinstance(df, pd.DataFrame)

    def test_dataarray_to_dataframe_no_warning_in_legacy(self):
        """Test that legacy mode suppresses warnings."""
        set_compat_mode('legacy')
        da = xr.DataArray([1, 2, 3], dims=['x'])

        import warnings
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            df = ensure_dataframe(da)

        # Filter for CompatibilityWarning only
        compat_warnings = [w for w in warning_list if issubclass(w.category, CompatibilityWarning)]
        assert len(compat_warnings) == 0
        set_compat_mode('auto')

    def test_dataarray_to_dataframe_strict_mode_raises(self):
        """Test that strict mode raises error on conversion."""
        set_compat_mode('strict')
        da = xr.DataArray([1, 2, 3], dims=['x'])

        with pytest.raises(TypeError, match="Expected pd.DataFrame"):
            df = ensure_dataframe(da)

        set_compat_mode('auto')

    def test_invalid_type_raises(self):
        """Test that unsupported type raises error."""
        with pytest.raises(TypeError, match="Cannot convert"):
            ensure_dataframe([1, 2, 3])


class TestEnsureDataArray:
    """Test ensure_dataarray function."""

    def test_dataarray_passthrough(self):
        """Test that DataArray passes through unchanged."""
        da = xr.DataArray([1, 2, 3], dims=['x'])
        result = ensure_dataarray(da)
        xr.testing.assert_equal(result, da)

    def test_dataframe_to_dataarray_explicit_params(self):
        """Test converting DataFrame to DataArray with explicit parameters."""
        set_compat_mode('legacy')
        df = pd.DataFrame({
            'x': ['a', 'b', 'c'],
            'y': [1, 2, 3],
            'values': [10, 20, 30]
        })

        da = ensure_dataarray(df, index_cols=['x', 'y'], value_col='values')

        assert isinstance(da, xr.DataArray)
        assert da.dims == ('x', 'y')
        assert len(da) == 3
        set_compat_mode('auto')

    def test_dataframe_to_dataarray_single_value_col(self):
        """Test converting DataFrame with single value column."""
        set_compat_mode('legacy')
        df = pd.DataFrame({
            'x': ['a', 'b', 'c'],
            'values': [1, 2, 3]
        })

        da = ensure_dataarray(df, index_cols=['x'])

        assert isinstance(da, xr.DataArray)
        assert 'x' in da.dims
        set_compat_mode('auto')

    def test_dataframe_to_dataarray_ambiguous_raises(self):
        """Test that DataFrame with multiple value columns raises error."""
        set_compat_mode('legacy')
        df = pd.DataFrame({
            'x': ['a', 'b', 'c'],
            'y': [1, 2, 3],
            'z': [4, 5, 6]
        })

        with pytest.raises(ValueError, match="multiple value columns"):
            ensure_dataarray(df, index_cols=['x'])

        set_compat_mode('auto')

    def test_dataset_to_dataarray_single_var(self):
        """Test converting Dataset with single variable to DataArray."""
        set_compat_mode('legacy')
        ds = xr.Dataset(
            {'temp': (['x'], [1, 2, 3])},
            coords={'x': ['a', 'b', 'c']}
        )

        da = ensure_dataarray(ds)

        assert isinstance(da, xr.DataArray)
        assert da.name == 'temp'
        set_compat_mode('auto')

    def test_dataset_to_dataarray_multiple_vars_raises(self):
        """Test that Dataset with multiple variables raises error."""
        set_compat_mode('legacy')
        ds = xr.Dataset(
            {
                'temp': (['x'], [1, 2, 3]),
                'humidity': (['x'], [4, 5, 6])
            },
            coords={'x': ['a', 'b', 'c']}
        )

        with pytest.raises(ValueError, match="multiple variables"):
            ensure_dataarray(ds)

        set_compat_mode('auto')

    def test_dataarray_with_warning(self):
        """Test that conversion emits warning in auto mode."""
        set_compat_mode('auto')
        df = pd.DataFrame({'x': ['a', 'b'], 'values': [1, 2]})

        with pytest.warns(CompatibilityWarning):
            da = ensure_dataarray(df, index_cols=['x'], value_col='values')

        assert isinstance(da, xr.DataArray)

    def test_invalid_type_raises(self):
        """Test that unsupported type raises error."""
        with pytest.raises(TypeError, match="Cannot convert"):
            ensure_dataarray([1, 2, 3])


class TestEnsureDataset:
    """Test ensure_dataset function."""

    def test_dataset_passthrough(self):
        """Test that Dataset passes through unchanged."""
        ds = xr.Dataset({'a': (['x'], [1, 2, 3])})
        result = ensure_dataset(ds)
        xr.testing.assert_equal(result, ds)

    def test_dataframe_to_dataset(self):
        """Test converting DataFrame to Dataset."""
        set_compat_mode('legacy')
        df = pd.DataFrame({
            'x': ['a', 'b'],
            'y': [1, 2],
            'z': [3, 4]
        })

        ds = ensure_dataset(df, index_cols=['x'])

        assert isinstance(ds, xr.Dataset)
        assert 'y' in ds.data_vars
        assert 'z' in ds.data_vars
        assert 'x' in ds.coords
        set_compat_mode('auto')

    def test_dataarray_to_dataset(self):
        """Test converting DataArray to Dataset."""
        set_compat_mode('legacy')
        da = xr.DataArray([1, 2, 3], dims=['x'], name='values')

        ds = ensure_dataset(da)

        assert isinstance(ds, xr.Dataset)
        assert 'values' in ds.data_vars
        set_compat_mode('auto')

    def test_invalid_type_raises(self):
        """Test that unsupported type raises error."""
        with pytest.raises(TypeError, match="Cannot convert"):
            ensure_dataset([1, 2, 3])


class TestGetUniqueValues:
    """Test get_unique_values function."""

    def test_dataframe_column(self):
        """Test getting unique values from DataFrame column."""
        df = pd.DataFrame({'ssp': ['SSP1', 'SSP2', 'SSP1', 'SSP3']})
        unique = get_unique_values(df, 'ssp')

        assert len(unique) == 3
        assert set(unique) == {'SSP1', 'SSP2', 'SSP3'}

    def test_dataframe_index(self):
        """Test getting unique values from DataFrame index."""
        df = pd.DataFrame({'value': [1, 2, 3]})
        df.index = pd.Index(['SSP1', 'SSP2', 'SSP1'], name='ssp')

        unique = get_unique_values(df, 'ssp')

        assert len(unique) == 2
        assert set(unique) == {'SSP1', 'SSP2'}

    def test_dataframe_multiindex(self):
        """Test getting unique values from MultiIndex DataFrame."""
        df = pd.DataFrame({'value': [1, 2, 3, 4]})
        df.index = pd.MultiIndex.from_tuples(
            [('SSP1', 2020), ('SSP2', 2020), ('SSP1', 2030), ('SSP3', 2020)],
            names=['ssp', 'year']
        )

        unique = get_unique_values(df, 'ssp')

        assert len(unique) == 3
        assert set(unique) == {'SSP1', 'SSP2', 'SSP3'}

    def test_dataset_coordinate(self):
        """Test getting unique values from Dataset coordinate."""
        ds = xr.Dataset(
            {'temp': (['ssp', 'year'], [[1, 2], [3, 4], [5, 6]])},
            coords={'ssp': ['SSP1', 'SSP2', 'SSP3'], 'year': [2020, 2030]}
        )

        unique = get_unique_values(ds, 'ssp')

        assert len(unique) == 3
        assert set(unique) == {'SSP1', 'SSP2', 'SSP3'}

    def test_dataarray_coordinate(self):
        """Test getting unique values from DataArray coordinate."""
        da = xr.DataArray(
            [[1, 2], [3, 4]],
            dims=['ssp', 'year'],
            coords={'ssp': ['SSP1', 'SSP2'], 'year': [2020, 2030]}
        )

        unique = get_unique_values(da, 'ssp')

        assert len(unique) == 2
        assert set(unique) == {'SSP1', 'SSP2'}

    def test_dataframe_missing_column_raises(self):
        """Test that missing column raises KeyError."""
        df = pd.DataFrame({'a': [1, 2, 3]})

        with pytest.raises(KeyError, match="Column 'missing' not found"):
            get_unique_values(df, 'missing')

    def test_dataset_missing_coordinate_raises(self):
        """Test that missing coordinate raises KeyError."""
        ds = xr.Dataset({'temp': (['x'], [1, 2, 3])})

        with pytest.raises(KeyError, match="Coordinate 'missing' not found"):
            get_unique_values(ds, 'missing')

    def test_unsupported_type_raises(self):
        """Test that unsupported type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported type"):
            get_unique_values([1, 2, 3], 'column')


class TestSafeIndexing:
    """Test safe_indexing function."""

    def test_dataframe_boolean_indexing(self):
        """Test boolean indexing on DataFrame."""
        df = pd.DataFrame({'ssp': ['SSP1', 'SSP2', 'SSP3'], 'value': [1, 2, 3]})
        result = safe_indexing(df, df['ssp'] == 'SSP1')

        assert len(result) == 1
        assert result['value'].iloc[0] == 1

    def test_dataset_boolean_indexing(self):
        """Test boolean indexing on Dataset."""
        ds = xr.Dataset(
            {'value': (['ssp'], [1, 2, 3])},
            coords={'ssp': ['SSP1', 'SSP2', 'SSP3']}
        )
        result = safe_indexing(ds, ds['ssp'] == 'SSP1')

        assert len(result['ssp']) == 1
        assert result['value'].values[0] == 1

    def test_dataarray_boolean_indexing(self):
        """Test boolean indexing on DataArray."""
        da = xr.DataArray(
            [1, 2, 3],
            dims=['ssp'],
            coords={'ssp': ['SSP1', 'SSP2', 'SSP3']}
        )
        result = safe_indexing(da, da.coords['ssp'] == 'SSP1')

        assert len(result) == 1
        assert result.values[0] == 1

    def test_unsupported_type_raises(self):
        """Test that unsupported type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported type"):
            safe_indexing([1, 2, 3], [True, False, True])


class TestSafeResetIndex:
    """Test safe_reset_index function."""

    def test_dataframe_reset_index(self):
        """Test resetting DataFrame index."""
        df = pd.DataFrame({'a': [1, 2]}, index=['x', 'y'])
        result = safe_reset_index(df)

        assert 'index' in result.columns
        assert len(result) == 2

    def test_dataarray_passthrough(self):
        """Test that DataArray passes through unchanged."""
        da = xr.DataArray([1, 2], dims=['x'])
        result = safe_reset_index(da)

        xr.testing.assert_equal(result, da)

    def test_dataset_passthrough(self):
        """Test that Dataset passes through unchanged."""
        ds = xr.Dataset({'a': (['x'], [1, 2])})
        result = safe_reset_index(ds)

        xr.testing.assert_equal(result, ds)


class TestSafeToDataFrame:
    """Test safe_to_dataframe function."""

    def test_converts_dataarray_no_warning(self):
        """Test that safe_to_dataframe doesn't emit warnings."""
        set_compat_mode('auto')
        da = xr.DataArray([1, 2, 3], dims=['x'], name='values')

        import warnings
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            df = safe_to_dataframe(da)

        # Should not emit CompatibilityWarning (warn=False)
        compat_warnings = [w for w in warning_list if issubclass(w.category, CompatibilityWarning)]
        assert len(compat_warnings) == 0
        assert isinstance(df, pd.DataFrame)

    def test_dataframe_passthrough(self):
        """Test that DataFrame passes through."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        result = safe_to_dataframe(df)

        pd.testing.assert_frame_equal(result, df)


class TestEnvironmentVariable:
    """Test environment variable override."""

    def test_env_var_overrides_mode(self):
        """Test that DSCIM_COMPAT_MODE environment variable works."""
        # Save original state
        original_mode = get_compat_mode()
        original_env = os.environ.get('DSCIM_COMPAT_MODE')

        try:
            # Set environment variable
            os.environ['DSCIM_COMPAT_MODE'] = 'legacy'
            set_compat_mode('auto')  # Set to auto, but env should override

            # Should not emit warning due to env var
            da = xr.DataArray([1, 2, 3], dims=['x'])
            import warnings
            with warnings.catch_warnings(record=True) as warning_list:
                warnings.simplefilter("always")
                df = ensure_dataframe(da)

            compat_warnings = [w for w in warning_list if issubclass(w.category, CompatibilityWarning)]
            assert len(compat_warnings) == 0

        finally:
            # Restore original state
            if original_env is None:
                os.environ.pop('DSCIM_COMPAT_MODE', None)
            else:
                os.environ['DSCIM_COMPAT_MODE'] = original_env
            set_compat_mode(original_mode)


class TestRoundtripConversions:
    """Test that conversions are lossless."""

    def test_dataframe_to_dataarray_to_dataframe(self):
        """Test DataFrame → DataArray → DataFrame roundtrip."""
        set_compat_mode('legacy')

        # Use a single dimension for cleaner roundtrip
        df_original = pd.DataFrame({
            'x': ['a', 'b', 'c'],
            'values': [10.5, 20.5, 30.5]
        })

        # Convert to DataArray
        da = ensure_dataarray(df_original, index_cols=['x'], value_col='values')

        # Convert back to DataFrame
        df_roundtrip = ensure_dataframe(da, name='values')

        # Check that data is preserved
        assert len(df_roundtrip) == 3
        assert 'values' in df_roundtrip.columns
        assert 'x' in df_roundtrip.columns
        # Check values are correct
        assert set(df_roundtrip['x'].values) == {'a', 'b', 'c'}
        np.testing.assert_array_almost_equal(sorted(df_roundtrip['values'].values), [10.5, 20.5, 30.5])

        set_compat_mode('auto')

    def test_dataarray_to_dataframe_to_dataarray(self):
        """Test DataArray → DataFrame → DataArray roundtrip."""
        set_compat_mode('legacy')

        da_original = xr.DataArray(
            [10, 20, 30],
            dims=['x'],
            coords={'x': ['a', 'b', 'c']},
            name='values'
        )

        # Convert to DataFrame
        df = ensure_dataframe(da_original, name='values')

        # Convert back to DataArray
        da_roundtrip = ensure_dataarray(df, index_cols=['x'], value_col='values')

        # Check that values are preserved
        assert len(da_roundtrip) == 3
        np.testing.assert_array_equal(da_roundtrip.values, da_original.values)

        set_compat_mode('auto')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
