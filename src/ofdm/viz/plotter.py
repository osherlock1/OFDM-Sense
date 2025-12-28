import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

def plot_symbol_freq(
        symbol: np.ndarray,
        title: str = "Frequency Domain Symbol",
        ax: Optional[plt.Axes] = None
):
    """
    Plots a Symbol in the Freq domain
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,4))
    
    n_symbol = len(symbol)
    k_neg = list(range(-(n_symbol // 2), 0))
    k_pos = list(range(1,(n_symbol // 2) + 1))
    k = np.array(k_neg + k_pos) #Combine and convert to ndarray

    ax.plot(k, np.real(symbol), label="Real")
    ax.plot(k, np.imag(symbol), label="Imag")

    ax.set_title(title)
    ax.set_xlabel("Freq Bins")
    ax.legend(loc='upper right')

    return ax