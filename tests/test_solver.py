import numpy as np
import pytest
from scipy import constants
from ofdm.simulation.solver import solve_tdoa, tdoa_cost_function, toa_cost_function


C = constants.c

@pytest.fixture
def four_rx_layout():
    rx_coords = np.array([
        [0.0, 0.0], # anchor
        [0.508, 0.137], # rx 2
        [0.0, 0.615], # rx 3
        [0.13, 0.22]
    ])
    known_tx = np.array([0.270, 0.970])
    return rx_coords, known_tx

def ideal_tdoa(tx_pos, rx_coords):
    distances = np.linalg.norm(rx_coords - tx_pos, axis=1)
    toa = distances / C
    return toa[1:] - toa[0]

class TestTDoACostFunction:
    def test_cost_function_zero_at_true_position(self, four_rx_layout):
        rx_coords, tx_true = four_rx_layout
        tdoas = ideal_tdoa(tx_true, rx_coords)
        residuals = tdoa_cost_function(tx_true, rx_coords, tdoas)
        np.testing.assert_allclose(residuals, 0.0, atol = 1e-9)

    def test_cost_function_residuals_shape(self, four_rx_layout):
        rx_coords, tx_true = four_rx_layout
        tdoas = ideal_tdoa(tx_true, rx_coords)
        residuals = tdoa_cost_function(tx_true, rx_coords, tdoas)
        assert residuals.shape == (len(rx_coords) - 1,)

    def test_cost_function_not_zero_at_different_position(self, four_rx_layout):
        rx_coords, tx_true = four_rx_layout
        wrong_guess = np.array([0.1, 0.1])
        tdoas = ideal_tdoa(tx_true, rx_coords)
        residuals = tdoa_cost_function(wrong_guess, rx_coords, tdoas)
        assert np.any(np.abs(residuals) > 1e-6)

class TestSolveTDoA:
    def test_solve_noiseless_recovers_true_position(self, four_rx_layout):
        rx_coords, tx_true = four_rx_layout
        tdoas = ideal_tdoa(tx_true, rx_coords)
        est = solve_tdoa(rx_coords, tdoas)
        assert est is not None
        np.testing.assert_allclose(est, tx_true, atol=1e-3)

    def test_with_default_initial_guess(self, four_rx_layout):
        rx_coords, tx_true = four_rx_layout
        tdoas = ideal_tdoa(tx_true, rx_coords)
        est = solve_tdoa(rx_coords, tdoas, initial_guess=None)
        assert est is not None
        np.testing.assert_allclose(est, tx_true, atol=1e-3)

    def solve_with_three_rx(self):
        rx_coords = np.array([
            [0.0, 0.0],
            [0.9, 0.0],
            [0.0, 0.9]
        ])
        tx_true = np.array([0.7, 0.4])
        tdoas = ideal_tdoa(tx_true, rx_coords)
        est = solve_tdoa(rx_coords, tdoas)
        assert est is not None
        np.testing.assert_allclose(est, tx_true, atol=1e-3)

    def solver_returns_ndarray(self, four_rx_layout):
        rx_coords, tx_true = four_rx_layout
        tdoas = ideal_tdoa(tx_true, rx_coords)
        est = solve_tdoa(rx_coords, tdoas)
        assert est is not None
        assert isinstance(est, np.ndarray)
        assert est.shape == (2,)