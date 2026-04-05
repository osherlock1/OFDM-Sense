import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from ofdm.simulation.monte_carlo import run_monte_carlo
from ofdm.config import loadLayout


def main():
    layout_config_pth = "./configs/layout.json"
    rx_coords, tx_true = loadLayout(layout_config_pth)

    rx_x = rx_coords[:, 0].reshape(-1, 1, 1)
    rx_y = rx_coords[:, 1].reshape(-1, 1, 1)

    x_range = np.linspace(0, 2, num=50)
    y_range = np.linspace(0, 2, num=50)
    X, Y = np.meshgrid(x_range, y_range)
    bounds = ([0, 2], [0, 2])
    error_heatmap = np.zeros(X.shape)

    for i in tqdm(range(len(x_range)), desc="Running Monte Carlo Simulations"):
        for j in range(len(y_range)):
            tx_coords = np.array([X[i][j], Y[i][j]])
            results = run_monte_carlo(
                rx_coords=rx_coords,
                tx_pos=tx_coords,
                sigma_ns=0.5,
                n_trials=30,
                seed=42,
                bounds=bounds
            )
            error_heatmap[i][j] = results['rmse']
    
    plt.figure(figsize=(10, 8))
    v_max = 1.5
    v_min = 0 
    cp = plt.pcolormesh(X, Y, error_heatmap,vmin=v_min, vmax=v_max, shading='auto', cmap='viridis')
    plt.colorbar(cp, label='RMSE (meters)')
    plt.scatter(rx_x, rx_y, marker='^', color='red', label='Receivers', s=150)
    plt.title('OFDM Localization Error Heatmap')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
