import numpy as np
from ofdm.simulation.geometry import compute_tdoa
from ofdm.simulation.noise_model import add_gausian_nosie
from ofdm.simulation.solver import solve_tdoa


def run_monte_carlo(tx_pos, rx_coords, sigma_ns, n_trials=1000, seed=None):
    """
    Run monte carlo TDOA localization simluation.
    Returns dict with estimates, errors, and summary stats
    """
    rng = np.random.default_rng(seed)
    ideal_tdoas = compute_tdoa(tx_pos, rx_coords)

    estimates = []
    for _ in range(n_trials):
        noisy_tdoas = add_gausian_nosie(ideal_tdoas, sigma_ns, rng=rng)
        est = solve_tdoa(rx_coords, noisy_tdoas)
        if est is not None:
            estimates.append(est)
    estimates = np.array(estimates)
    errors = np.linalg.norm(estimates - tx_pos, axis=1)
    return {
        "estimates": estimates,
        "errors": errors,
        "rmse": np.sqrt(np.mean(errors**2)),
        "mean_error": np.mean(errors),
        "p95_error": np.percentile(errors, 95),
        "centroid": np.mean(estimates, axis=0),
        "n_converged": len(estimates),
        "n_trials": n_trials,
    }