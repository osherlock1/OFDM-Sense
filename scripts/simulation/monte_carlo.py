import argparse
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tx", nargs=2, type=float, default = [0.565, 0.906])
    parser.add_argument("--sigma-ns", type=float, default=0.05)
    parser.add_argument("--trials", type=int, default=1000)
    args = parser.parse_args()

    rx_coords = np.array([
        [0.0, 0.0],
        [0.508, 0.137],
        [0.0, 0.615],
        [0.13, 0.22]
    ])

    results = run_monte_carlo(
        tx_pos=np.array(args.tx),
        rx_coords=rx_coords,
        sigma_ns=args.sigma_ns,
        n_trials=args.trials,
        seed=42,
    )

    print(f"Converged:  {results['n_converged']}/{results['n_trials']}")
    print(f"RMSE:       {results['rmse']*100:.2f} cm")
    print(f"Mean error: {results['mean_error']*100:.2f} cm")
    print(f"P95 errors: {results['p95_error']*100:.2f} cm")
    print(f"Centroid:   X={results['centroid'][0]:.4f} cm, Y={results['centroid'][1]:.4f} cm")

if __name__ == "__main__":
    main()