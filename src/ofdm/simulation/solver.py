import numpy as np
from scipy import constants
from scipy.optimize import least_squares

C = constants.c

def tdoa_cost_function(guess, rx_coords, delay_diffs_sec):
    """
    Calculates residuals for TDOA least-squares solve.                                            
    rx_coords[0] must be the anchor receiver.                                                     
    delay_diffs_sec: array of length len(rx_coords)-1, t_i - t_anchor in seconds.
    """
    x, y = guess
    residuals = np.zeros(len(rx_coords) - 1)

    anchor_x, anchor_y = rx_coords[0]
    dist_to_anchor = np.sqrt((x - anchor_x)**2 + (y - anchor_y)**2)

    for i in range(1, len(rx_coords)):
        rx_x, rx_y = rx_coords[i]
        dist_to_rx = np.sqrt((x - rx_x)**2 + (y - rx_y)**2)
        theoretical_diff = dist_to_rx - dist_to_anchor
        measured_diff = delay_diffs_sec[i-1] * C
        residuals[i-1] = theoretical_diff - measured_diff
    
    return residuals

def solve_tdoa(rx_coords, delay_diffs_sec, initial_guess=None):
    """                                                        
    Runs LM least-squares TDOA solve for a single set of delay measurements.
    Returns (x, y) estimate or None if solve failed.                                              
    """
    
    if initial_guess is None:
        initial_guess = np.mean(rx_coords, axis=0) # centroid of recievers
    
    result = least_squares(
        tdoa_cost_function,
        initial_guess,
        args=(rx_coords, delay_diffs_sec),
        method='lm'
    )

    if result.success:
        return result.x
    return None


def toa_cost_function(guess, rx_coords, measured_distance):
    """
    PLACEHOLDER
    """
    x, y = guess
    residuals = np.zeros(len(rx_coords))

    for i in range(len(rx_coords)):
        rx_x, rx_y = rx_coords[i]  
        theoretical_dist = np.sqrt((x - rx_x)**2 + (y-rx_y)**2)
        residuals[i] = theoretical_dist - measured_distance
    return residuals