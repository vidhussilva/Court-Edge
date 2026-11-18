import numpy as np
import pytest

from courtedge.market.devig import devig, devig_power, implied, overround


def test_multiplicative_sums_to_one():
    pa = devig([1.5, 2.0, 3.4], [2.9, 2.04, 1.38])
    pb = devig([2.9, 2.04, 1.38], [1.5, 2.0, 3.4])
    np.testing.assert_allclose(pa + pb, 1.0)


def test_fair_book_unchanged():
    assert devig(2.0, 2.0) == pytest.approx(0.5)
    assert devig(1.25, 5.0) == pytest.approx(0.8)  # 0.8 + 0.2 = 1 already


def test_hand_calculated_overround():
    # 1/1.8 + 1/2.1 = 0.5556 + 0.4762 = 1.0317
    assert overround(1.8, 2.1) == pytest.approx(0.031746, abs=1e-6)
    assert devig(1.8, 2.1) == pytest.approx(0.5556 / 1.0317, abs=1e-4)


def test_power_shades_longshot_more():
    fav_mult = devig(1.2, 5.5)
    fav_pow = devig_power(1.2, 5.5)
    assert fav_pow > fav_mult  # longshot absorbs more of the overround
    ia, ib = implied(1.2), implied(5.5)
    k = np.log(fav_pow) / np.log(ia)
    assert ia**k + ib**k == pytest.approx(1.0, abs=1e-9)


def test_power_vectorised_and_symmetric():
    a = np.array([1.5, 2.0, 4.0])
    b = np.array([2.7, 2.0, 1.28])
    np.testing.assert_allclose(devig_power(a, b) + devig_power(b, a), 1.0, atol=1e-9)


def test_invalid_odds():
    with pytest.raises(ValueError):
        implied([1.0, 2.0])
    with pytest.raises(ValueError):
        devig(2.0, 2.0, method="shin")
