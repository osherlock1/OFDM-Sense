import numpy as np
from scipy.optimize import least_squares
from scipy import constants
import json
import pandas as pd
import os
import matplotlib.pyplot as plt

C = constants.c
DELAY_DATA_PATH = "./metadata/delay_calc.json"

RX_COORDS = np.array([
    [0.0, 0.0], # RX 1
    [0.3937, 0.1422], # RX 2
    [0.1397, 0.5207], # RX 3
])

TX_TRUE = np.array([0.5613, 0.7264])

def load_tdoa_from_csvs():
    base_dir = "./experiments/unpacked_data"

    df_rx2_ch0 = pd.read_csv(os.path.join(base_dir, "rx2_channel0.csv"))
    df_rx2_ch1 = pd.read_csv(os.path.join(base_dir, "rx2_channel1.csv"))
    
    df_rx3_ch0 = pd.read_csv(os.path.join(base_dir, "rx3_channel0.csv"))
    df_rx3_ch1 = pd.read_csv(os.path.join(base_dir, "rx3_channel1.csv"))

    col_name = 'delay0'

    anchor_t1_ns = df_rx2_ch0[col_name].values[0]
    rover_t1_ns = df_rx2_ch1[col_name].values[0]

    anchor_t2_ns = df_rx3_ch0[col_name].values[0]
    rover_t2_ns = df_rx3_ch1[col_name].values[0]

    # calculate TDoA in nanoseconds
    dt_1_ns = rover_t1_ns - anchor_t1_ns
    dt_2_ns = rover_t2_ns - anchor_t2_ns

    # convert to seconds
    dt_1_sec = dt_1_ns * 1e-9
    dt_2_sec = dt_2_ns * 1e-9 

    return np.array([dt_1_sec, dt_2_sec])


def main():
    # rx coordinates
    
    try:
        delay_diffs_sec = load_tdoa_from_csvs()
        print(f"TDoA 1 (RX2_Anhor): {delay_diffs_sec[0]*1e9:.3f} ns")
        print(f"TDoA (RX-Anchor): {delay_diffs_sec[1]*1e9:.3f} ns")
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        return
    
    print("\n--- Running TDoA Solver ---")

    initial_guess = np.array([0.3,0.3])

    result = least_squares(
        tdoa_cost_fuction,
        initial_guess,
        args=(RX_COORDS, delay_diffs_sec),
        method='lm'
    )

    est_x, est_y = result.x
    error_m = np.sqrt((est_x - TX_TRUE[0])**2 + (est_y - TX_TRUE[1])**2)

    print(f"Solver Converged: {result.success}")
    print(f"Estimated TX Position: X: {est_x:.4f}m, Y: {est_y:.4f}m")
    print(f"Ground Truth TX:       X: {TX_TRUE[0]:.4f}m, Y: {TX_TRUE[1]:.4f}m")
    print(f"Localization Error:    {error_m * 100:.2f} cm") 

    # --- Plotting the Desk Layout ---
    plt.figure(figsize=(8, 8))
    
    # Plot Receivers
    plt.scatter(RX_COORDS[0,0], RX_COORDS[0,1], c='blue', marker='^', s=150, label='RX1 (Anchor)')
    plt.scatter(RX_COORDS[1:,0], RX_COORDS[1:,1], c='cyan', marker='^', s=100, label='RX2/RX3 (Rover)')
    
    # Plot Transmitter Data
    plt.scatter(TX_TRUE[0], TX_TRUE[1], c='green', marker='s', s=100, label='TX (Ground Truth)')
    plt.scatter(est_x, est_y, c='red', marker='x', s=150, linewidths=3, label='TX (Estimated)')

    # Formatting
    plt.title("OFDM-Sense mmWave Localization Results")
    plt.xlabel("X Coordinate (meters)")
    plt.ylabel("Y Coordinate (meters)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.axis('equal') 
    plt.show()





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