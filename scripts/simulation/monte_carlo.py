import argparse
import numpy as np 
import matplotlib.pyplot as plt
from ofdm.viz.sim_plotter import plot_mc_results, plot_tdoa_hyperbolas, DraggableSimulation
from ofdm.simulation.monte_carlo import run_monte_carlo
from ofdm.config import loadLayout

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tx", nargs=2, type=float)
    parser.add_argument("--sigma-ns", type=float, default=0.05)
    parser.add_argument("--trials", type=int, default=1000)
    args = parser.parse_args()

    layout_conf_path = "./configs/layout.json"
    rx_coords, tx_true = loadLayout(layout_conf_path)

    if args.tx is not None:
        tx_true = np.array(args.tx)

    bounds = ([0, 2], [0, 2])

    results = run_monte_carlo(
        tx_pos=tx_true,
        rx_coords=rx_coords,
        sigma_ns=args.sigma_ns,
        n_trials=args.trials,
        seed=42,
        bounds=bounds
    )

    print(f"Converged:  {results['n_converged']}/{results['n_trials']}")
    print(f"RMSE:       {results['rmse']*100:.2f} cm")
    print(f"Mean error: {results['mean_error']*100:.2f} cm")
    print(f"P95 errors: {results['p95_error']*100:.2f} cm")
    print(f"Centroid:   X={results['centroid'][0]:.4f} cm, Y={results['centroid'][1]:.4f} cm")

    ax = plot_mc_results(results, tx_true, rx_coords, args.sigma_ns)
    plot_tdoa_hyperbolas(tx_true, rx_coords, results, ax)
    plt.tight_layout()
    sim = DraggableSimulation(
        ax=ax, 
        tx_pos=tx_true,
        rx_coords=rx_coords,
        sigma_ns=args.sigma_ns,
        n_trials=args.trials
    )
    
    plt.show()

if __name__ == "__main__":
    main()