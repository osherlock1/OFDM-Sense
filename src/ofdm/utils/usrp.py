import subprocess
import sys
from dataclasses import dataclass
import os
import time
import signal
import yaml
#Get USRP Configs from YAML Config file


@dataclass
class USRPConfig:
    """Configuration of URSP Hardware"""
    build_path: str = "./build/TXRX_FROM_FILE"
    tx_addr: str = "addr=192.168.30.2"
    rx_addr: str = "addr=192.168.30.2"
    tx_rate: float = 100e6 #TX sampling rate
    rx_rate: float = 100e6 #RX sampling rate
    tx_freq: float = 60e6 # Digital up conversion center frequency tx
    rx_freq: float = 60e6 # Digital up conversion center frequency rx

    subdev: str = "A:0"
    tx_channel_idx: str = "0"
    rx_channel_idx: str = "0"

    wave_type:str = "SINE"
    wave_freq: float = "100e3"
    ampl: float = 0.3
    tx_gain: float = 0
    rx_gain:float = 0
    otw:str = "sc16"
    type:str = "float"
    ref:str = "external"

def load_config(path: str) -> USRPConfig:
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)

    config = USRPConfig(**config_dict)

    print("--- USRP Configuration Loaded ---")
    print(config)
    print("---------------------------------")

    return config

def run_transfer(
        config: USRPConfig,
        tx_file: str,
        rx_file: str,
        nsamps: int,
):
    """
    Execute the C++ USRP driver via subprocess.
    """
    
    #Build command list
    cmd = [
        config.build_path,
        "--tx-args", config.tx_addr,
        "--rx-args", config.rx_addr,
        "--tx-rate", str(config.tx_rate),
        "--rx-rate", str(config.rx_rate),
        "--tx-freq", str(config.tx_freq),
        "--rx-freq", str(config.rx_freq),
        "--tx-gain", str(config.tx_gain),
        "--rx-gain", str(config.rx_gain),

        #File I/O
        "--file", rx_file,
        "--tx-file", tx_file,
        "--nsamps", str(nsamps),

        "--tx-channels", config.tx_channel_idx,
        "--rx-channels", config.rx_channel_idx,


        "--ref", config.ref,

        #Channel Configuration
        "--rx-subdev", config.subdev,
        "--tx-subdev", config.subdev,


        #Fixed constants (OTW format, types)
        "--otw", "sc16",
        "--type", "float",
        "--tx-type", "float",
        "--settling", "0",
        "--tx-spb", "0",
        "--tx-repeat", "false",
    ]

    print(f"\n[USRP] Executing: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print("[USRP] Transfer Complete!\n")
    except subprocess.CalledProcessError as e:  
        print(f"[Error] USRP driver failed with exit code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"[Error] Could not find executable at {config.build_path}")
        sys.exit(1)



# ---------------------------
# NOT CURRENTLY NEEDED
# --------------------------


# def run_transfer_sep(        
#         config: USRPConfig,
#         tx_file: str,
#         rx_file: str,
#         nsamps: int,
#         channel: str = "B",
# ):
#     """  
#     Run transfer using seperate RX and TX subprocesses
#     """   

#     subdev = "A:0" if channel == "A" else "B:0"

#     #run RX
#     #Build command list
#     rx_cmd = [
#         config.rx_build_path,
#         "--args", config.rx_addr,
#         "--rate", str(config.rx_rate),
#         "--freq", str(config.rx_freq),
#         "--gain", str(config.rx_gain),
#         "--file", rx_file,
#         "--nsamps", str(nsamps),
#         "--subdev", subdev,
#         "--channels", "0",
#         "--otw", "sc16",
#         "--type", "float",
#     ]

#     print(f"[USRP] Executing RX: {' '.join(rx_cmd)}")

#     with open(f"rx_log.txt", 'w') as outfile:
#         rx_process = subprocess.Popen(rx_cmd, stdout=outfile, stderr=outfile)
#     print(["[USRP] Waiting 3 seconds for RX to lock..."])
#     time.sleep(3)

#     #Run tx
#     print("[USRP] executing TX burst...")
#     run_tx(config=config, tx_file=tx_file, channel=channel)

#     #Wait
#     time.sleep(1)

#     #Stop RX procress
#     print("[USRP] Stopping RX Process...")
#     rx_process.send_signal(signal.SIGINT)
#     try:
#         rx_process.wait(timeout=2)
#     except subprocess.TimeoutExpired:
#         rx_process.kill()

#     print("[USRP] Transfer Complete!")


# def run_rx(
#         config:USRPConfig,
#         rx_file:str,
#         nsamps: int,
#         channel: str = "B"

# ):
#     """  
#     Run RX from the C++ USRP driver
#     """

#     #Channel Selection
#     subdev = "A:0" if channel == "A" else "B:0"

#     #Build command list
#     cmd = [
#         config.rx_build_path,
#         "--args", config.rx_addr,
#         "--rate", str(config.rx_rate),
#         "--freq", str(config.rx_freq),
#         "--gain", str(config.rx_gain),
#         "--file", rx_file,
#         "--nsamps", str(nsamps),
#         "--subdev", subdev,
#         "--channels", "0",
#         "--otw", "sc16",
#         "--type", "float",
#     ]

#     print(f"[USRP] Executing RX: {' '.join(cmd)}")

#     try:
#         subprocess.run(cmd, check=True)
#     except subprocess.CalledProcessError as e:
#         print(f"[Error] USRP RX Driver failed returncode: {e.returncode}")
#         sys.exit(1)
#     except FileNotFoundError:
#         print(f"[Error] RX failed could not find driver: {config.rx_build_path}")
#         sys.exit(1)



# def run_tx(
#         config: USRPConfig,
#         tx_file: str,
#         channel: str = "B",

        
# ):
#     """  
#     Send a TX burst through the USRP by reading data from a file
#     """

#     #Channel Selection
#     subdev = "A:0" if channel == "A" else "B:0"

#     #Build command list
#     cmd = [
#         config.tx_build_path,
#         "--args", config.tx_addr,
#         "--rate", str(config.tx_rate),
#         "--freq", str(config.tx_freq),
#         "--gain", str(config.tx_gain),
#         "--file", tx_file,
#         "--subdev", subdev,
#         "--channels", "0",
#         "--otw", "sc16",
#         "--type", "float",
#         "--spb", "10000", #FIXME THIS IS A TEMP VALUE IT MIGHT NEED ADJUSTING OR AUTO ADJUSTING
#     ]

#     print(f"[USRP] Executing TX: {' '.join(cmd)}")

#     try:
#         subprocess.run(cmd, check=True)
#         print(f"[USRP] TX Complete!")
#     except subprocess.CalledProcessError as e:
#         print(f"[Error] TX failed with error code: {e.returncode}")
#         sys.exit(1)
#     except FileNotFoundError:
#         print(f"[Error] File Not Found: {config.tx_build_path}")
#         sys.exit(1)