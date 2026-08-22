import numpy as np
import pytest
from scipy import constants
from ofdm.simulation.geometry import compute_toa, compute_tdoa

C = constants.c


@pytest.fixture
def four_rx_layout():
    rx_coords = np.array(
        [
            [0.0, 0.0],  # anchor
            [0.508, 0.137],  # rx 2
            [0.0, 0.615],  # rx 3
            [0.13, 0.22],
        ]
    )
    known_tx = np.array([0.270, 0.970])
    return rx_coords, known_tx


def test_toa_single_receiver():
    """TOA should equal distance / c."""
    tx = np.array([1.0, 0.0])
    rx = np.array([[0.0, 0.0]])
    toa = compute_toa(tx, rx)
    np.testing.assert_allclose(toa[0], 1.0 / C, rtol=1e-9)


def test_tdoa_zero_when_equidistant():
    """TX at center of symmetric layout should produce all-zero TDOAs."""
    rx_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    tx = np.array([0.5, 0.5])  # equidistant from all four
    tdoa = compute_tdoa(tx, rx_coords)
    np.testing.assert_allclose(tdoa, 0.0, atol=1e-12)


def test_tdoa_output_shape(four_rx_layout):
    rx_coords, tx = four_rx_layout
    tdoa = compute_tdoa(tx, rx_coords)
    assert tdoa.shape == (len(rx_coords) - 1,)
