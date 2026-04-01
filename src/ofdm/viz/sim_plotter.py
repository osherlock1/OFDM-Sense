import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from typing import Optional
from scipy import constants
from ofdm.simulation.geometry import compute_tdoa
from ofdm.simulation.monte_carlo import run_monte_carlo

C = constants.c

def plot_mc_results(
        results: dict,
        tx_pos: np.ndarray,
        rx_coords: np.ndarray,
        sigma_ns: float,
        ax: Optional[plt.Axes] = None,
):
    """
    Plot Monte Carlo localizaiton results.
    results: dict returned by run_monte_carlo()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))

    estimates = results["estimates"]
    centroid = results["centroid"]

    ax.scatter(estimates[:, 0], estimates[:, 1],
               c='red', marker='o', s=20, alpha=0.3, label='Estimates', zorder=2)
    
    # centroid
    ax.scatter(*centroid, c='purple', marker='X', s=200, edgecolor='black', linewidths=1.5, label='Centroid', zorder=5)

    # tx positions
    ax.scatter(*tx_pos, c='green', marker='s', s=150, label='TX (Ground Truth)', zorder=5)

    # rx positions
    ax.scatter(*rx_coords[0], c='blue', marker='^', s=150, label='Anchor RX', zorder=4)
    ax.scatter(rx_coords[1:, 0], rx_coords[1:, 1], c='cyan', marker='^', s= 100, label='Roaming RX', zorder=4)

    stats = (
        f"$\\sigma$ = {sigma_ns} ns\n"
        f"RMSE      = {results['rmse']*100:.2f} cm\n"
        f"Mean err  = {results['mean_error']*100:.2f} cm\n"
        f"P95 err   = {results['p95_error']*100:2.2f} cm\n"
        f"Converged: = {results['n_converged']}/{results['n_trials']}"
    )
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=9, 
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title(f"OFDM Monte Carlo Simulation ({results['n_trials']})")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_aspect('equal')

    return ax

def plot_tdoa_hyperbolas(tx_pos, rx_coords, results, ax, sigma_ns=None):
    """
    For each non-anchor RX, plot the hyperbola defined by the ideal TDOA
    """

    ideal_tdoas = compute_tdoa(tx_pos, rx_coords)

    x = np.linspace(-0.5, 1.5, 800)
    y = np.linspace(-0.5, 1.5, 800)
    X, Y = np.meshgrid(x, y)

    anchor = rx_coords[0]
    dist_to_anchor = np.sqrt((X - anchor[0])**2 + (Y - anchor[1])**2)

    for i in range(1, len(rx_coords)):
        rx = rx_coords[i]
        dist_to_rx = np.sqrt((X - rx[0])**2 + (Y - rx[1])**2)
        diff = dist_to_rx - dist_to_anchor - (ideal_tdoas[i - 1] * C)
        ax.contour(X, Y, diff, levels=[0], linewidths=1.5, linestyles='-')


class DraggableSimulation:
    def __init__(self, ax, tx_pos, rx_coords, sigma_ns, n_trials=1000):
        self.ax = ax
        self.tx_pos = tx_pos.copy()
        self.rx_coords = rx_coords.copy()
        self.sigma_ns = sigma_ns
        self.n_trials = n_trials
        self._dragging = None
        self._pick_radius = 0.05
        fig = ax.get_figure()
        fig.canvas.mpl_connect('button_press_event', self._on_press)
        fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        fig.canvas.mpl_connect('button_release_event', self._on_release)
    
    def _hit(self, event, pos):
        if event.xdata is None:
            return False
        return np.hypot(event.xdata - pos[0], event.ydata - pos[1]) < self._pick_radius
    
    def _on_press(self, event):
        if self._hit(event, self.tx_pos):
            self._dragging = ('tx', None)
        else:
            for i, rx in enumerate(self.rx_coords):
                if self._hit(event, rx):
                    self._dragging = ('rx', i)
                    break

    def _on_motion(self, event):
        if self._dragging is None or event.xdata is None:
            return
        kind, idx = self._dragging
        if kind == 'tx':
            self.tx_pos[:] = [event.xdata, event.ydata]
        else:
            self.rx_coords[idx] = [event.xdata, event.ydata]
        self._redraw()

    def _on_release(self, event):
        self._dragging = None
    
    def _redraw(self):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.cla()
        results = run_monte_carlo(self.tx_pos, self.rx_coords, self.sigma_ns, self.n_trials, seed=42)
        plot_mc_results(results, self.tx_pos, self.rx_coords, self.sigma_ns, ax=self.ax)
        plot_tdoa_hyperbolas(self.tx_pos, self.rx_coords, results, self.ax)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.get_figure().canvas.draw_idle()

