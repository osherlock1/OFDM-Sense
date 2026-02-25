import numpy as np
from scipy.optimize import least_squares
from scipy import constants
import json

C = constants.c
DELAY_DATA_PATH = "./metadata/delay_calc.json"

def main():
    # rx coordinates
    rx_coords = np.array([
        [0.0, 0.0], # RX 1
        [2.0, 0.0], # RX 2
        [2.0, 2.0], # RX 3
        [0.0, 2.0] # RX 4
    ])

    with open(DELAY_DATA_PATH, 'r') as f:
        delay_data = json.load(f)


    delays_sec = delay_data['delays']
    delays_dist = delay_data['raw_distance']

    initial_guess = np.array([1.0, 1.0])

    # run non-linear least squares solver
    result = least_squares(
        toa_cost_function,
        initial_guess,
        args=(rx_coords, delays_dist)
    )

    # extract the 2D coordinate and error metric
    estimated_tx_pos = result.x
    success = result.success
    cost = result.cost

    print("--- Localization Results ---")
    print(f"Solver Converged: {success}")
    print(f"Estimated TX Position: X: {estimated_tx_pos[0]:.3f}m, Y: {estimated_tx_pos[1]:.3f}m")
    print(f"Residula Cost (Error): {cost:.5f}")



def toa_cost_function(guess, rx_coords, measured_distance):
    """
    Calculates difference between theoretical distances based on guessed (x,y) and the actual measured distnace.

    """

    x, y = guess
    residuals = np.zeros(len(rx_coords))

    for i in range(len(rx_coords)):
        rx_x, rx_y = rx_coords[i]

        theoretical_dist = np.sqrt((x - rx_x)**2 + (y - rx_y)**2)

        residuals[i] = theoretical_dist - measured_distance[i]

    return residuals

def tdoa_cost_fuction(guess, rx_coords, delay_diffs_sec):
    """
    Calculates residuals for TDoA
    rx_coords[0] MUST be anchor rx
    delay_diffs_sec is 3 element array: [t2-t1, t3-t1, t4-t1]
    """

    x, y = guess

    residuals = np.zeros(len(rx_coords) - 1)

    # calc theoretical distance from anchor rx
    anchor_x, anchor_y = rx_coords[0]
    dist_to_anchor = np.sqrt((x - anchor_x)**2 + (y - anchor_y)**2)

    # loop through remaining 3 receivers
    for i in range(1, len(rx_coords)):
        rx_x, rx_y = rx_coords[i]

        dist_to_rx = np.sqrt((x - rx_x)**2 + (y - rx_y)**2)
        theoretical_diff = dist_to_rx - dist_to_anchor
        measured_diff = delay_diffs_sec[i-1] * C

        residuals[i-1] = theoretical_diff - measured_diff

    return residuals

if __name__ == "__main__":
    main()