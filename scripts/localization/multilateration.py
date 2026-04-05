import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from ofdm.simulation.solver import solve_tdoa
from ofdm.viz.sim_plotter import plot_experiment_results, plot_tdoa_hyperbolas

def load_rover_position(roaming_positions_path: Path) ->np.ndarray:
    with open(roaming_positions_path) as f:
        data = json.load(f)
    pos = next(iter(data.values()))
    return np.array([pos["x"], pos["y"]])

def main():
    parser = argparse.ArgumentParser(description="Run TDOA multilateration on a processed experiment.")
    parser.add_argument("--experiment", type=str, required=True, help ="Path to experiment directory")
    parser.add_argument("--devices", nargs="+", required=True, help="Device archive names e.g. RX3ch1 RX5ch1, RX7ch1")
    parser.add_argument("--anchor", type=str, required=True, help="Anchor device key in fixed_positions.json e.g ANCHORch0")
    parser.add_argument("--bias-ns", type=float, default=3.661, help="Hardware TDOA bias to subtract (nanoseconds)")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment)

    with open(experiment_dir / "fixed_positions.json") as f:
        fixed = json.load(f)

    anchor_pos = np.array([fixed[args.anchor]["x"], fixed[args.anchor]["y"]])
    tx_true = np.array([fixed["TX"]["x"], fixed["TX"]["y"]])

    rover_positions = []
    device_dataframes = []

    for device in args.devices:
        archive_dir = experiment_dir / f"{device}_archive"
        rover_positions.append(load_rover_position(archive_dir / "roaming_positions.json"))

        df = pd.read_csv(archive_dir / "delays.csv")
        device_dataframes.append(df[["run", "delay0", "delay1"]].rename(columns={"delay0": f"delay0_{device}", "delay1": f"delay1_{device}"}))

    rx_coords = np.array([anchor_pos] + rover_positions)

    print(f"ROVER POSITIONS {rover_positions}")

    merged = device_dataframes[0][["run"]].copy()
    for df in device_dataframes:
        merged = merged.merge(df, on="run")

    estimated_positions = []

    MAX_TDOA_NS = 30

    for _, row in merged.iterrows():
        delay_diffs_s = []
        valid=True
        for device in args.devices:
            tdoa_ns = row[f"delay0_{device}"] - row[f"delay1_{device}"] - args.bias_ns
            if abs(tdoa_ns) > MAX_TDOA_NS:
                valid = False
                break
            delay_diffs_s.append(tdoa_ns * 1e-9)
        if not valid:
            continue
        result = solve_tdoa(rx_coords, np.array(delay_diffs_s))
        if result is not None:
            estimated_positions.append(result)

    estimated_positions = np.array(estimated_positions)
    n_converged = len(estimated_positions)
    n_total = len(merged)

    if n_converged == 0:
        print("No successful solves.")
        return
    
    centroid = np.mean(estimated_positions, axis=0)
    errors = np.linalg.norm(estimated_positions - tx_true, axis=1)
    centroid_error = np.linalg.norm(centroid - tx_true)

    print(f"Converged:      {n_converged}/{n_total}")
    print(f"Centroid:       X={centroid[0]:.4f}m  Y={centroid[1]:.4f}m")
    print(f"Mean error:     {np.mean(errors)*100:.2f} cm")
    print(f"RMSE:           {np.sqrt(np.mean(errors**2))*100:.2f} cm")
    print(f"P95 error:      {np.percentile(errors, 95)*100:.2f} cm")
    print(f"Centroid error: {centroid_error*100:.2f} cm")

    results = {                                                                          
        "estimates": estimated_positions,                                                
        "centroid": centroid,
        "rmse": np.sqrt(np.mean(errors**2)),
        "mean_error": np.mean(errors),
        "p95_error": np.percentile(errors, 95),                                          
        "centroid_error": centroid_error,
        "n_converged": n_converged,                                                      
        "n_trials": n_total,
    } 

    fig, ax = plt.subplots(figsize=(9, 9))
    plot_experiment_results(results, tx_true, rx_coords, ax=ax)
    plot_tdoa_hyperbolas(tx_true, rx_coords, results, ax)
    ax.set_title(f"OFDM Localization")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()