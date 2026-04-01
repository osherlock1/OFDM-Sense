import numpy as np
import matplotlib.pyplot as plt
from ofdm.simulation.monte_carlo import run_monte_carlo


def main():
    rx_coords = np.array([
        [0.0, 0.0],
        [0.508, 0.137],
        [0.0, 0.615],
        #[0.0, 1.0]
    ])
    rx_x = rx_coords[:, 0].reshape(-1, 1, 1)
    rx_y = rx_coords[:, 1].reshape(-1, 1, 1)

    x_range = np.linspace(0, 1, num=40)
    y_range = np.linspace(0, 1, num=40)
    X, Y = np.meshgrid(x_range, y_range)

    error_heatmap = np.zeros(X.shape)

    for i in range(len(x_range)):
        for j in range(len(y_range)):
            tx_coords = np.array([X[i][j], Y[i][j]])
            results = run_monte_carlo(
                rx_coords=rx_coords,
                tx_pos=tx_coords,
                sigma_ns=0.1,
                n_trials=10,
                seed=42,
            )
            error_heatmap[i][j] = results['rmse']
    
    plt.figure(figsize=(10, 8))
    v_max = 1  
    v_min = 0 
    cp = plt.pcolormesh(X, Y, error_heatmap,vmin=v_min, vmax=v_max, shading='auto', cmap='viridis')
    plt.colorbar(cp, label='RMSE (meters)')
    plt.scatter(rx_x, rx_y, marker='^', color='red', label='Receivers')
    plt.title('OFDM Localization Error Heatmap')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()