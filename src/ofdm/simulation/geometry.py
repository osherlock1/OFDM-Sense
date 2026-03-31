import numpy as np
from scipy import constants

C = constants.c

def ideal_tdoa(tx_pos, rx_coords):
    distances = np.linalg.norm(rx_coords - tx_pos, axis=1)
    toa = distances / C
    return toa[1:] - toa[0]