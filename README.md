![Project Banner](banner.png)

![Python](https://img.shields.io/badge/language-Python-blue)
![C++](https://img.shields.io/badge/language-C++-00599C)
![Status](https://img.shields.io/badge/Status-In_Development-yellow)
![Publication](https://img.shields.io/badge/Publication-IEEE_ORSS_2026-blue)

# OFDM-Sense

OFDM-Sense is a Joint Communication and Sensing (JCAS) research platform developed as the 2025/26 URI ELECOMP Capstone project under Dr. Guoyi Xu. It implements an OFDM transceiver on a USRP X310 software-defined radio (SDR) to explore the use of OFDM waveforms for simultaneous wireless communication and device localization via time-difference-of-arrival (TDOA).

## Publication

Our paper, **"2D Localization Leveraging OFDM Signals,"** has been accepted for publication in the 2026 IEEE ORSS proceedings.

> O'Malley Sherlock\*, Royaljohn Southamavong\*, and Guoyi Xu, "2D Localization Leveraging OFDM Signals," in *2026 IEEE ORSS*, 2026. (to appear)


A link will be added once available.

## Table of Contents
- [Publication](#publication)
- [How it Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Workflow](#workflow)
- [Localization Experiments](#localization-experiments)
- [Simulation](#simulation)
- [Project Structure](#project-structure)

---

## How it Works

A transmitter sends an OFDM packet (preamble + data symbols) from a USRP X310. One or more receivers capture the signal. The receiver pipeline:

1. **Synchronization** Schmidl-Cox algorithm detects packet start and estimates coarse CFO
2. **Channel estimation** pilot symbols estimate frequency-domain channel response
3. **Equalization & demodulation** 16-QAM symbols are recovered and evaluated (EVM, BER, SER)
4. **Delay estimation** matched filter + sub-sample interpolation measures propagation delay
5. **Localization** TDOA across multiple receivers feeds a least-squares multilateration solver

---

## Prerequisites

### Hardware
- [USRP X310](https://www.ettus.com/all-products/x310-kit/) 
- External 10 MHz reference clock (shared across all devices for synchronization)
- 10 GbE connection per USRP

### Software
- Python >= 3.8
- [UHD (USRP Hardware Driver)](https://github.com/EttusResearch/uhd)  required to build the C++ control binary
- CMake >= 3.8 and a C++14 compiler

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/osherlock1/OFDM-Sense.git
cd OFDM-Sense
```

### 2. Build the C++ USRP control binary
```bash
mkdir build && cd build
cmake ..
make
cd ..
```
This produces `build/TXRX_FROM_FILE`, which the Python scripts call via subprocess.

### 3. Install the Python package
```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Verify the installation
```bash
pytest
```

---

## Configuration

### USRP Hardware (`configs/usrp_settings.yaml`)
Copy and edit this file to match your hardware setup:

```yaml
build_path: "./build/TXRX_FROM_FILE"
tx_addr: "addr0=192.168.30.2,addr1=192.168.40.2"
rx_addr: "addr0=192.168.30.2,addr1=192.168.40.2"

subdev: "A:0 B:0"
tx_channel_idx: "1"
rx_channel_idx: "0,1,2"

tx_rate: 100e6
rx_rate: 100e6
tx_freq: 60e6
rx_freq: 60e6

tx_gain: 0
rx_gain: 0
ref: "external"   # external 10 MHz reference clock
```

Update `tx_addr` / `rx_addr` to match your USRP IP addresses. The `ref: "external"` field requires a shared 10 MHz clock source connected to all devices.

---

## Workflow

### Quick start (simulation only - no hardware needed)
```bash
# Generate a synthetic OFDM packet with noise
python scripts/generate_packet.py --n_symb 30 --snr 20

# Unpack and evaluate
python scripts/unpack_rx.py --sim --plot
```

### Hardware transfer
```bash
# 1. Generate the transmit packet
python scripts/generate_packet.py --n_symb 30

# 2. Verify the USRP is reachable (sends a sine wave and checks CFO)
python scripts/test_sin_wave.py

# 3. Run the transfer
python scripts/run_transfer.py

# 4. Unpack and evaluate the received signal
python scripts/unpack_rx.py --plot
```

---

## Localization Experiments

See [`scripts/experiment_scripts/README.md`](scripts/experiment_scripts/README.md) for a full tutorial. The short version:

```bash
# 1. Edit EXPERIMENT_NAME, ROAMING_DEVICES, FIXED_DEVICES at the top of collect_raw_data.py

# 2. Collect data (prompts for device positions)
python scripts/experiment_scripts/collect_raw_data.py --runs 5 --experiments_dir ./experiments

# 3. Process raw .dat files into CSV
python scripts/experiment_scripts/process_experiment.py --experiment_pth ./experiments/my_experiment

# 4. Run multilateration
python scripts/localization/multilateration.py --experiment ./experiments/my_experiment --devices RX3ch1 --anchor ANCHORch0
```

---

## Simulation

Run a Monte Carlo TDOA localization simulation without any hardware:

```bash
python scripts/simulation/monte_carlo.py --sigma-ns 0.1 --trials 1000
```

The simulation uses an interactive drag-and-drop UI to reposition TX and RX nodes and recompute localization error in real time.

---

## Project Structure

```
OFDM-Sense/
├── src/ofdm/               # Installable Python library
│   ├── core/               # Waveform, preamble, payload construction
│   ├── channel/            # Channel estimation, CFO correction, delay
│   ├── modulation/         # 16-QAM
│   ├── processing/         # RX pipeline, batch processing
│   ├── simulation/         # TDOA geometry, solver, Monte Carlo
│   ├── utils/              # USRP config, data generation, evaluation
│   ├── viz/                # Plotting utilities
│   └── config.py           # OFDMConfig dataclass (N=256, CP=64, Fs=100MHz)
├── scripts/                # Runnable entry-point scripts
│   ├── experiment_scripts/ # Data collection and processing workflows
│   ├── localization/       # Multilateration
│   ├── simulation/         # Monte Carlo simulation
│   ├── delay/              # Delay estimation and calibration
│   └── image_demo/         # Image transmission demo
├── usrp_control_files/     # C++ UHD driver source
├── tests/                  # pytest test suite
├── configs/                # YAML hardware configuration
├── data_files/             # Reference packet data
├── notebooks/              # Analysis notebooks
└── CMakeLists.txt          # C++ build system
```
