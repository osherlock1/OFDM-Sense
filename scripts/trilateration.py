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
    [0.0, 0.0], # ANchor
    [0.4699, 0.08255], # RX 1
    [0.53975, 0.14605], # RX 2
    [0.5461, 0.3048] # rx 3
])

TX_TRUE = np.array([0.565, 0.906])

def load_tdoa_from_csvs():
    base_dir = "./experiments/multilat_data1"
    
    df_rx2_ch0 = pd.read_csv(os.path.join(base_dir, "rx1_channel0.csv"))
    df_rx2_ch1 = pd.read_csv(os.path.join(base_dir, "rx1_channel1.csv"))
    
    df_rx3_ch0 = pd.read_csv(os.path.join(base_dir, "rx2_channel0.csv"))
    df_rx3_ch1 = pd.read_csv(os.path.join(base_dir, "rx2_channel1.csv"))

    df_rx4_ch0 = pd.read_csv(os.path.join(base_dir, "rx3_channel0.csv"))
    df_rx4_ch1 = pd.read_csv(os.path.join(base_dir, "rx3_channel1.csv"))

    col_name = 'delay0'


    raw_anchor_t1_ns = df_rx2_ch1[col_name].values # Ch 1
    raw_rover_t1_ns  = df_rx2_ch0[col_name].values # Ch 0
    raw_anchor_t2_ns = df_rx3_ch1[col_name].values # Ch 1
    raw_rover_t2_ns  = df_rx3_ch0[col_name].values # Ch 0
    raw_anchor_t3_ns = df_rx4_ch0[col_name].values # Ch 1
    raw_rover_t3_ns = df_rx4_ch1[col_name].values # Ch 0

    # calc tdoa
    raw_dt_1_ns = raw_rover_t1_ns - raw_anchor_t1_ns
    raw_dt_2_ns = raw_rover_t2_ns - raw_anchor_t2_ns
    raw_dt_3_ns = raw_rover_t3_ns - raw_anchor_t3_ns

    # correct hardware delays
    HARDWARE_BIAS_NS = 3.463
    bias_corrected_dt_1 = raw_dt_1_ns - HARDWARE_BIAS_NS
    bias_corrected_dt_2 = raw_dt_2_ns - HARDWARE_BIAS_NS
    bias_corrected_dt_3 = raw_dt_3_ns - HARDWARE_BIAS_NS

    # correct SDR clock slip
    dt_1_ns = (bias_corrected_dt_1 + 5.0) % 10.0 - 5.0
    dt_2_ns = (bias_corrected_dt_2 + 5.0) % 10.0 - 5.0
    dt_3_ns = (bias_corrected_dt_3 + 5.0) % 10.0 - 5.0

    # convert to seconds
    dt_1_sec = dt_1_ns * 1e-9
    dt_2_sec = dt_2_ns * 1e-9
    dt_3_sec = dt_3_ns * 1e-9

    return np.column_stack((dt_1_sec, dt_2_sec, dt_3_sec))



def main():
    try:
        all_delay_diffs = load_tdoa_from_csvs()
        num_packets = len(all_delay_diffs)
        print(f"--- Loaded {num_packets} packets for processing ---")
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        return
    
    print("\n--- Running TDoA Solver ---")

    initial_guess = np.array([0.3, 0.3])
    estimated_positions = []
    errors = []

    for i in range(num_packets):
        delay_diffs_sec = all_delay_diffs[i]

        result = least_squares(
            tdoa_cost_fuction,
            initial_guess,
            args=(RX_COORDS, delay_diffs_sec),
            method='lm'
        )

        if result.success:
            est_x, est_y = result.x
            estimated_positions.append([est_x, est_y])
            error_m = np.sqrt((est_x - TX_TRUE[0])**2 + (est_y - TX_TRUE[1])**2)
            errors.append(error_m)
    
    estimated_positions = np.array(estimated_positions)
    errors = np.array(errors)

    centroid_x, centroid_y = np.mean(estimated_positions, axis=0)
    mean_error = np.mean(errors)
    centroid_error = np.sqrt((centroid_x - TX_TRUE[0])**2 + (centroid_y - TX_TRUE[1])**2)

    print(f"Successful Solves:     {len(estimated_positions)}/{num_packets}")
    print(f"Centroid TX Position:  X: {centroid_x:.4f}m, Y: {centroid_y:.4f}m")
    print(f"Mean Packet Error:     {mean_error * 100:.2f} cm")
    print(f"Centroid Error:        {centroid_error * 100:.2f} cm")


    plt.figure(figsize=(9, 9))
    plt.scatter(RX_COORDS[0,0], RX_COORDS[0,1], c='blue', marker='^', s=150, label='RX1 (Anchor)', zorder=3)
    plt.scatter(RX_COORDS[1:,0], RX_COORDS[1:,1], c='cyan', marker='^', s=100, label='RX2/RX3 (Rover)', zorder=3)
    
    # truth
    plt.scatter(TX_TRUE[0], TX_TRUE[1], c='green', marker='s', s=150, label='TX (Ground Truth)', zorder=3)

    # estimates
    plt.scatter(estimated_positions[:, 0], estimated_positions[:, 1], 
                c='red', marker='o', s=30, alpha=0.3, label='Individual Estimates', zorder=2)
    
    # centroid
    plt.scatter(centroid_x, centroid_y, c='purple', marker='X', s=200, edgecolor='black', 
                linewidths=2, label='TX (Centroid)', zorder=4)

    plt.title(f"OFDM-Sense Localization Cluster ({len(estimated_positions)} packets)")
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

    # loop through remaining receivers
    for i in range(1, len(rx_coords)):
        rx_x, rx_y = rx_coords[i]

        dist_to_rx = np.sqrt((x - rx_x)**2 + (y - rx_y)**2)
        theoretical_diff = dist_to_rx - dist_to_anchor
        measured_diff = delay_diffs_sec[i-1] * C

        residuals[i-1] = theoretical_diff - measured_diff

    return residuals

if __name__ == "__main__":
    main()