import numpy as np
import scipy.signal
from typing import Tuple, Optional
from ofdm.config import OFDMConfig

#Temp debugg
from ofdm.viz import plotter
import matplotlib.pyplot as plt

def calculate_schmidl_cox_metrics(rx_signal: np.ndarray, config:OFDMConfig)->Tuple[np.ndarray, np.ndarray]:
    """  
    Calculate the M and P metrics packet dection and syncronization
    """

    L = config.N // 2

    #Energy Calc
    #R is moving sum of energy of the second half of the window
    energy = np.abs(rx_signal) ** 2

    R = scipy.signal.fftconvolve(energy, np.ones(L), mode='valid')

    #Correlate the signal with a version of itself shifted by L
    r_upper = rx_signal[L:]
    r_lower = rx_signal[:-L]

    #Multiply elemeent wise
    mult_term = np.conj(r_lower * r_upper)

    #Moving sum over window L
    P = scipy.signal.fftconvolve(mult_term, np.ones(L), mode='valid')

    #Compute M Metric
    min_len = min(len(P), len(R))
    P = P[:min_len]
    R = R[:min_len]

    #Avoid division by zero
    R[R == 0] = 1e-10

    #M = |P|^2 / R^2
    M = (np.abs(P) ** 2) / (R ** 2)

    return M, P
    #Plot original siganl
    # plotter.plot_time_series(signal=rx_signal, title="Original Signal")
    # plotter.plot_time_series(signal=energy, title="Energy")
    # plotter.plot_time_series(signal=R, title="R")
    # plotter.plot_time_series(signal=P, title="P")
    # plotter.plot_time_series(signal=M, title="M")
    # plt.show()