import numpy as np


def add_nosie(signal: np.ndarray, snr_db: float, seed:int = None) -> np.ndarray:
    """
    Adds Gaussian Noise to a signal
    """

    rng = np.random.default_rng(seed)

    #Calculate Power
    Px = np.mean(np.abs(signal) ** 2)
    
    #Convert SNR from dB to linear
    snr_lin = 10 ** (snr_db / 10.0)

    #Calculate noise variance
    sigma2 = Px / snr_lin

    #Generate complex noise
    noise = (rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)) * np.sqrt(sigma2/2)

    return signal + noise                                                                                             