# Scripts

## Top-level

### eval_rx.py
Evaluates the performance of each subcarrier in the demodulated symbol.

### generate_packet.py
Generates OFDM packets for transfer. Specify number of data symbols, add Gaussian noise for testing, and set the output file name.

### plot_rx_signal.py
Visualizes the generated and received raw time-series signals from a USRP transfer.

### run_transfer.py
Sends generated data through the USRP system. Specify USRP parameters and the communication channel.

### test_sin_wave.py
Quickly checks if the USRP platform is set up correctly and successfully transferring data. Also calculates carrier frequency offset (CFO).

### unpack_rx.py
Unpacks received OFDM packets, evaluates performance (EVM/BER/SER), and saves the demodulated data to a JSON file. Use `--sim` to unpack directly from a generated TX file with no hardware capture needed.

### verify_sync_usrp.py
Tests the Schmidl-Cox synchronization algorithm's performance.

### trilateration.py — *legacy*
Early standalone TDOA trilateration script with hardcoded RX coordinates. Superseded by [`localization/multilateration.py`](localization/multilateration.py).

---

## delay/

### mf_calibrate.py
Calibrates the matched-filter delay estimate against a wired reference, storing constants to `metadata/calibration.json`.

### calc_delay.py
Computes per-channel propagation delay from received packets using the matched-filter calibration. Requires `mf_calibrate.py` to have been run first.

### mf_test.py
Test script for getting a distance estimate from matched-filter (MF) magnitude outputs.

### run_delay_experiment.py
Runs the transfer + delay-calculation pipeline repeatedly to collect delay measurements over multiple runs.

---

## localization/

### multilateration.py
Runs TDOA multilateration on a processed experiment to estimate 2D position.

---

## simulation/

### monte_carlo.py
Runs a Monte Carlo TDOA localization simulation with an interactive drag-and-drop UI for repositioning TX/RX nodes.

### heatmap.py
Sweeps TX position over a grid, running a Monte Carlo simulation at each point to produce a localization-error heatmap.

---

## experiment_scripts/

See [`experiment_scripts/README.md`](experiment_scripts/README.md) for the full data-collection tutorial.

### collect_raw_data.py
Collects raw RX capture data for a localization experiment, prompting for device positions and saving runs into an experiment directory.

### process_experiment.py
Processes raw `.dat` archives from an experiment directory into delay/performance data.

### unpack_experiment.py — *legacy*
Early experiment-unpacking helper. Superseded by `process_experiment.py`.
