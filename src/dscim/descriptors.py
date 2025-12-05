# from https://github.com/dssg/dickens/blob/master/src/descriptors.py
import os


class cachedproperty:
    """Non-data descriptor decorator implementing a read-only property
    which overrides itself on the instance with an entry in the
    instance's data dictionary, caching the result of the decorated
    property method.
    """

    def __init__(self, func):
        self.__func__ = func

    def __get__(self, instance, _type=None):
        if instance is None:
            return self

        # Use a private attribute name to store the raw cached value
        cache_attr = f'_{self.__func__.__name__}_cached'

        # Check if we already have a cached value
        if not hasattr(instance, cache_attr):
            # Compute and cache the raw value
            setattr(instance, cache_attr, self.__func__(instance))

        # Get the cached value
        value = getattr(instance, cache_attr)

        # Apply legacy mode conversions if needed (every time, not just first time)
        if os.environ.get('DSCIM_COMPAT_MODE', '').lower() == 'legacy':
            import xarray as xr

            # damage_function_points: Convert to DataFrame and sum over regions
            if self.__func__.__name__ == 'damage_function_points':
                from dscim.utils.compat import ensure_dataframe
                if isinstance(value, xr.Dataset):
                    # Sum over region dimension if it exists
                    if 'region' in value.dims:
                        value = value.sum(dim='region')
                    return ensure_dataframe(value, warn=False)

            # global_consumption: Sum over regions for DataArrays
            elif self.__func__.__name__ == 'global_consumption':
                if isinstance(value, xr.DataArray) and 'region' in value.dims:
                    value = value.sum(dim='region')

        return value
