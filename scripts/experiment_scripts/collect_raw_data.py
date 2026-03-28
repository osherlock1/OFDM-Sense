import subprocess
import os
import shutil
import os
from datetime import datetime
import argparse
from ofdm.utils import usrp
import json

USRP_CONFIG_PATH = "./configs/usrp_settings.yaml"
SOURCE_DAT_FILE = "./data_files/rand_ofdm_packet_rx"

# --------- MODIFY --------------
EXPERIMENT_NAME = "virtual_multilateration_3" # CHOOSE NAME OF EXPERIMENT TO BE RUN
ROAMING_DEVICES = ["RX2ch1"] # NAME OF DEVICE THAT IS MOVED (WILL ASK FOR POSITIONS EACH RUN)
FIXED_DEVICES = ["ANCHORch0", "TX"] # NAME OF DEVICES THAT ARE FIXED (WILL ONLY ASK ONCE PER EXPERIMENT)
# -------------------------------

RUN_NAME = "".join(ROAMING_DEVICES) + "_"
EXPERIMENT_DIR = f"/home/guoyixu/OFDM_Sense/EXPERIMENTS/{EXPERIMENT_NAME}"
DESTINATION_DIR = EXPERIMENT_DIR + f"/{RUN_NAME}archive/"

#Load Configurations
usrp_conf = usrp.load_config(USRP_CONFIG_PATH)
rx_channel_idx = usrp_conf.rx_channel_idx.replace(",", "")

def parse_measurement(prompt):
    """Parse a measurement string like '5in', '30cm', '1.2m' and return value in meters."""
    while True:
        value_str = input(prompt).strip().lower()

        if value_str.endswith("in"):
            return float(value_str[:-2]) * 0.0254
        elif value_str.endswith("cm"):
            return float(value_str[:-2]) * 0.01
        elif value_str.endswith("m"):
            return float(value_str[:-1])
        else:
            print("Please specify a valid unit")


def get_coordinate_input():
    """Takes in a string from a user in inches, cm, or m and returns the value in m"""
    x = parse_measurement((f"  X:  "))
    y = parse_measurement((f"  Y:  "))
    return {"x": round(x, 4), "y": round(y, 4)}

def get_fixed_positions(experiment_dir, fixed_devices):
    """Save the positions of the fixed TX AND RX MODULES"""
    fixed_path = os.path.join(experiment_dir, "fixed_positions.json")

    if os.path.exists(fixed_path): #if positions are already saved
        with open(fixed_path, "r") as f:
            positions = json.load(f)
        print(f"Loaded fixed positions from {fixed_path}")
        for device, pos in positions.items():
            print(f" {device}: x={pos['x']}, y={pos['y']}")
        return positions
    
    positions = {}
    for device in fixed_devices: #no fixed positions saved
        print(f"Enter fixed {device} position (Specify unit like 30.2cm, 3m, 2in):")
        positions[device] = get_coordinate_input()
    
    with open(fixed_path, "w") as f:
        json.dump(positions, f, indent=2)
    print(f"Saved fixed positions to {fixed_path}")
    return positions

def get_roaming_positions(roaming_rx_devices, run_dir):
    """Save the positions of the roaming RX or TX Modules"""
    roaming_path = os.path.join(run_dir, "roaming_positions.json")

    positions = {}
    for device in roaming_rx_devices:
        print(f"Enter roaming {device} position (Specify unit like 30.2cm, 3m, 2in):")
        positions[device] = get_coordinate_input()
    
    with open(roaming_path, "w") as f:
        json.dump(positions, f, indent=2)
    print(f"Saved roaming positions to {roaming_path}")
    return positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1, help="specify number of time experiement will run")
    args = parser.parse_args()

    if not os.path.exists(DESTINATION_DIR):
        os.makedirs(DESTINATION_DIR)
        print(f"Created archive directory: {DESTINATION_DIR}")

    get_fixed_positions(EXPERIMENT_DIR, fixed_devices=FIXED_DEVICES)
    get_roaming_positions(run_dir=DESTINATION_DIR, roaming_rx_devices=ROAMING_DEVICES)

    for run in range(args.runs):
        print(f"\n========== RUN {run+1}/{args.runs} ==========\n")
        print("--- Starting Transfer (REF) ----")
        subprocess.run(["python", "./scripts/run_transfer.py"], check=True)
        print("--- Transfer Complete ---")
        print("---Unpacking OFDM Data---")
        subprocess.run(["python", "./scripts/unpack_rx.py"], check=True)
        print("---Finished Unpacking---")

        for channel in rx_channel_idx:
            source_dat_file = f"{SOURCE_DAT_FILE}.0{channel}.dat"
            channel_file = f"{DESTINATION_DIR}channel{channel}/"

            if not os.path.exists(channel_file):
                os.makedirs(channel_file)
                print(f"Created archive directory: {channel_file}")

            if os.path.exists(source_dat_file):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_filename = f"{EXPERIMENT_NAME}_run_{run+1}_{timestamp}.dat"
                destination_path = os.path.join(channel_file, new_filename)
                shutil.move(source_dat_file, destination_path)
                print(f"Saved run data to: {destination_path}")

            else:
                print(f"WARNING: Expected data file {source_dat_file} not found.")

if __name__ == "__main__":
    main()