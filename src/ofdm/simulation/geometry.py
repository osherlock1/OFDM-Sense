import numpy as np
from scipy import constants

C = constants.c


def compute_toa(tx_pos, rx_coords):
    distances = np.linalg.norm(rx_coords - tx_pos, axis=1)
    return distances / C


def compute_tdoa(tx_pos, rx_coords):
    toa = compute_toa(tx_pos, rx_coords)
    return toa[1:] - toa[0]  # roaming rx - anchor


def compute_distances(tx_pos, rx_coords):
    return np.linalg.norm(rx_coords - tx_pos, axis=1)
