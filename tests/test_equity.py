import pandas
import xarray as xr
from pandas.testing import assert_frame_equal
from xarray.testing import assert_allclose
import pytest

from . import open_zipped_results
from dscim.menu.equity import EquityRecipe


@pytest.mark.parametrize("menu_class", [EquityRecipe], indirect=True)
def test_equity_points(menu_instance, discount_types):
    path = f"equity_{discount_types}_eta{menu_instance.eta}_rho{menu_instance.rho}_damage_function_points.csv"
    # Set index so can compare unsorted tables.
    idx = ["model", "ssp", "rcp", "gcm", "year"]
    expected = open_zipped_results(path).set_index(idx)
    actual = menu_instance.damage_function_points.set_index(idx)
    # rtol and atol are 10x the largest deviation from this fixture measured with
    # numpy's AVX512 code paths. atol is 7.6e-4 of the median absolute damage (2.6e10).
    assert_frame_equal(
        expected,
        actual,
        rtol=3e-4,
        atol=2e7,
        check_like=True,
    )


@pytest.mark.parametrize("menu_class", [EquityRecipe], indirect=True)
def test_equity_coefficients(menu_instance, discount_types):
    path = f"equity_{discount_types}_eta{menu_instance.eta}_rho{menu_instance.rho}_damage_function_coefficients.nc4"
    expected = open_zipped_results(path)
    actual = menu_instance.damage_function_coefficients
    # rtol and atol are 10x the largest deviation from this fixture measured with
    # numpy's AVX512 code paths. atol is 9.6e-6 to 1.2e-5 of the median absolute
    # coefficient (6.6e10 to 8.3e10).
    assert_allclose(
        expected.transpose(*sorted(expected.dims)).sortby(list(expected.dims)),
        actual.transpose(*sorted(actual.dims)).sortby(list(actual.dims)),
        rtol=5e-6,
        atol=8e5,
    )


@pytest.mark.parametrize("menu_class", [EquityRecipe], indirect=True)
def test_equity_fit(menu_instance, discount_types):
    path = f"equity_{discount_types}_eta{menu_instance.eta}_rho{menu_instance.rho}_damage_function_fit.nc4"
    expected = open_zipped_results(path)
    actual = menu_instance.damage_function_fit
    # rtol and atol are 10x the largest deviation from this fixture measured with
    # numpy's AVX512 code paths. atol is 4.1e-5 of the median absolute fitted value
    # (4.9e12).
    assert_allclose(
        expected.transpose(*sorted(expected.dims)).sortby(list(expected.dims)),
        actual.transpose(*sorted(actual.dims)).sortby(list(actual.dims)),
        rtol=3e-5,
        atol=2e8,
    )


@pytest.mark.parametrize("menu_class", [EquityRecipe], indirect=True)
def test_equity_global_consumption(menu_instance, discount_types):
    path = f"equity_{discount_types}_eta{menu_instance.eta}_rho{menu_instance.rho}_global_consumption.nc4"
    expected = open_zipped_results(path)
    actual = menu_instance.global_consumption.squeeze()
    # Small format hack from I/O
    if isinstance(expected, xr.Dataset):
        expected = expected.to_array().squeeze().drop("variable")

    assert_allclose(
        expected.transpose(*sorted(expected.dims)).sortby(list(expected.dims)),
        actual.transpose(*sorted(actual.dims)).sortby(list(actual.dims)),
    )


@pytest.mark.parametrize("menu_class", [EquityRecipe], indirect=True)
def test_equity_scc(menu_instance, discount_types):
    path = (
        f"equity_{discount_types}_eta{menu_instance.eta}_rho{menu_instance.rho}_scc.nc4"
    )
    expected = open_zipped_results(path)
    actual = menu_instance.calculate_scc.squeeze()

    # Small format hack from I/O
    if isinstance(expected, xr.Dataset):
        expected = expected.to_array().squeeze().drop("variable")

    assert_allclose(
        expected.transpose(*sorted(expected.dims)).sortby(list(expected.dims)),
        actual.transpose(*sorted(actual.dims)).sortby(list(actual.dims)),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize("discount_types", ["euler_ramsey"], indirect=True)
@pytest.mark.parametrize("menu_class", [EquityRecipe], indirect=True)
def test_global_damages_calculation(menu_instance):
    global_damages = menu_instance.global_damages_calculation()
    assert (
        isinstance(global_damages, pandas.DataFrame)
        and "region" not in global_damages.columns
    )
