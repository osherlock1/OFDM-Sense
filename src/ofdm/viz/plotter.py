import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def plot_symbol_freq(
    symbol: np.ndarray,
    title: str = "Frequency Domain Symbol",
    ax: Optional[plt.Axes] = None,
):
    """
    Plots a Symbol in the Freq domain
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    n_symbol = len(symbol)
    k_neg = list(range(-(n_symbol // 2), 0))
    k_pos = list(range(1, (n_symbol // 2) + 1))
    k = np.array(k_neg + k_pos)  # Combine and convert to ndarray

    ax.stem(k, np.real(symbol), label="Real")
    ax.stem(k, np.imag(symbol), label="Imag")

    ax.set_title(title)
    ax.set_xlabel("Freq Bins")
    ax.legend(loc="upper right")

    return ax


def plot_time_series(
    signal: np.ndarray,
    fs: float = 1.0,
    ax: Optional[plt.Axes] = None,
    title: str = "Time Domain Signal",
):
    """
    Plots raw IQ sampels over time
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    t = np.arange(len(signal)) / fs

    ax.plot(t, np.real(signal), label="Real")
    ax.plot(t, np.imag(signal), label="Imag")

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right")

    return ax


def plot_constellation(
    iq_data: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Constellation Plot",
):
    """
    Plots RX IQ salmpes as QAM16 Format
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(np.real(iq_data), np.imag(iq_data), s=10, alpha=0.5)

    # 16-QAM reference grid
    levels = [-3, -1, 1, 3]
    for x in levels:
        for y in levels:
            ax.plot(x, y, "r+", markersize=10, markeredgewidth=1.5)

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel("In-Phase")
    ax.set_ylabel("Quadrature")
    ax.set_aspect("equal")

    return ax
