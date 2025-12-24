import subprocess, pathlib, shlex, sys, time
import numpy as np
import os, sys, numpy as np
import matplotlib.pyplot as plt
from ofdm_manager import OFDMManager
from ofdm_symbol import OFDMSymbol
from subcarrier_map import SubcarrierMap
from data_generator import DataGenerator
import scipy
from scipy.interpolate import interp1d
import json
from pilot_symbol import PilotSymbol
import argparse
from sync_symbol import SyncSymbol
from dataclasses import dataclass

#------Global Class Def ----------
#Helper Classes Instantiation
map = SubcarrierMap()
om = OFDMManager()
dg = DataGenerator()


@dataclass
class FileConfig:
    TX_FILE_PATH = "data_files/rand_ofdm_packet.dat"
    RX_FILE_PATH = "data_files/rand_ofdm_packet_rx.dat"
    REF_FILE_PATH = "data_files/rand_ofdm_packet_ref.json"

@dataclass
class SymbolConfig:
    with open(FileConfig.REF_FILE_PATH, 'r') as file:
        data = json.load(file)
        N_data_symbols = data["N_data_symbols"]
        n_samples = data["n_samples"]
        i_samples = data["i_data"]
        q_samples = data["q_data"]
    N_PILOT_SYMBOLS = 1
    N_SYNC_SYMBOLS = 1

    #Index Values
    pilot_k = map.pilots_k
    pilots_idx = np.array([map.idx(k) for k in map.pilots_k])
    data_k = map.data_bins
    data_idx = np.array([map.idx(k) for k in map.data_bins])

@dataclass
class FreqConfig:
    FS = 100e6
    n_bins = 2 ** 14
    f_grid = np.linspace(-FS / 2, FS / 2, n_bins, endpoint=False)
    n_pilot = np.arange(map.N) / FS

def main():


    #---------CLI ARGS---------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', type=bool, default = False)
    parser.add_argument('--channel', '-c', type=str, default = "B")
    args = parser.parse_args()
    SIMULATION = args.sim
    CHANNEL_SEL = args.channel

    SAMPLE_BUFFER = 500 #Number of additional samples to transfer from the length of the OFDM packet
    FS = 100e6
    #CPP ARGS
    BUILD_PATH = "./build/TXRX_FROM_FILE"
    TX_ADDR = "addr=192.168.30.2"
    RX_ADDR = "addr=192.168.30.2"
    TX_RATE = str(FS)
    RX_RATE = str(FS)
    TX_FREQ = "60e6"
    RX_FREQ = "60e6"
    WAVE_TYPE = "SINE"
    WAVE_FREQ = "100e3"
    AMPL = "0.3"
    TX_GAIN = "0"
    RX_GAIN = "0"
    OTW = "sc16"
    TYPE = "float"
    FILE_NAME = FileConfig.RX_FILE_PATH
    NSAMPLES = str(SymbolConfig.n_samples + SAMPLE_BUFFER)
    SETTLING = "0"
    TX_FILE = FileConfig.TX_FILE_PATH
    #TX_FILE = "data_files/ofdm_iq_interleaved.dat" #NAME OF FILE YOU WANT TO READ AND SEND
    TX_TYPE = "float"
    TX_SPB = "0"
    TX_REPEAT = "false"
    RX_CHANNEL = "0" #Specify number of active channel (0 = one channel, 1 = both (2) channels)
    TX_CHANNEL = "0"


    if CHANNEL_SEL == "A":
        print("CHANNEL SELECTED A:0")
        RX_SUBDEV = "A:0"
        TX_SUBDEV = "A:0"
    elif CHANNEL_SEL == "B":
        print("CHANNEL SELECTED B:0")
        RX_SUBDEV = "B:0"
        TX_SUBDEV = "B:0"
    else:
        print(f"WARNING: INVALID CHANNEL SELCTION {CHANNEL_SEL}.  SETTING CHANNEL TO B:0")
        RX_SUBDEV = "B:0"
        TX_SUBDEV = "B:0"

    #Build the command to run
    run_cmd = [
        BUILD_PATH,
        "--tx-args", TX_ADDR,
        "--rx-args", RX_ADDR,
        "--tx-rate", TX_RATE,
        "--rx-rate", RX_RATE,
        "--tx-freq", TX_FREQ,
        "--rx-freq", RX_FREQ,
    #   "--wave-type", WAVE_TYPE,
    #   "--wave-freq", WAVE_FREQ,
    #   "--ampl", AMPL,
        "--tx-gain", TX_GAIN,
        "--rx-gain", RX_GAIN,
        "--otw", OTW,
        "--type", TYPE,
        "--file", FILE_NAME,
        "--nsamps", NSAMPLES,
        "--settling", SETTLING,
        "--tx-file", TX_FILE,
        "--tx-type", TX_TYPE,
        "--tx-spb", TX_SPB,
        "--tx-repeat", TX_REPEAT,
        "--rx-channels", RX_CHANNEL,
        "--tx-channels", TX_CHANNEL,
        "--rx-subdev", RX_SUBDEV,
        "--tx-subdev", TX_SUBDEV
    ]

    #----- Check for Simluation ------
    if SIMULATION is False:
        print("\n")
        print(f"Running {BUILD_PATH}...")
        print(str(run_cmd))
        subprocess.run(run_cmd)
        print(f"Run of {BUILD_PATH} complete!")
        file_name = FileConfig.RX_FILE_PATH
    else:
        print("Simulating OFDM Transfer... \n")
        file_name = FileConfig.TX_FILE_PATH

    #------- Unpack Reciever IQ sampels---------
    print("Unpacking OFDM Symbol....")
    iq = np.fromfile(file_name, dtype = np.complex64)

    #-------- Synq ------------------
    print("Calculating M Values ....")
    M_Values, P_Values = om.calc_M_values(iq)
    print("Done!")
    
    M_filtered = om.filter_M(M_Values) #Filter M for clear peak sync idx
    sync_idx = np.argmax(M_filtered)
    start_idx = np.array([sync_idx])[0]

    #----------Unpack Symbols ----------
    packet_len = (map.N + map.cp_len) * (SymbolConfig.N_data_symbols + SymbolConfig.N_PILOT_SYMBOLS + SymbolConfig.N_SYNC_SYMBOLS)
    print(f"Packet Length = {packet_len}")
    ofdm_packet = iq[start_idx : start_idx + packet_len] #Extract all Symbols
    print(f"Length of ofdm_packet {len(ofdm_packet)}")
    print(f"Length of split {SymbolConfig.N_data_symbols + SymbolConfig.N_PILOT_SYMBOLS + SymbolConfig.N_SYNC_SYMBOLS}")
    symbols = np.split(ofdm_packet, SymbolConfig.N_data_symbols + SymbolConfig.N_PILOT_SYMBOLS + SymbolConfig.N_SYNC_SYMBOLS)

    symbols_no_cp = [symbol[:-map.cp_len] for symbol in symbols]


    data_symbols = symbols_no_cp[SymbolConfig.N_SYNC_SYMBOLS + SymbolConfig.N_PILOT_SYMBOLS:]
    #FIXME: HARDCODED VALUES
    sync_symbol = symbols[0]
    pilot_symbol = symbols[1]

    #-----------Pilot Symbol CFO-------------
    pilot_ref_t = om.create_tx_block(PilotSymbol())
    pilot_ref_k = PilotSymbol().symbol

    print(f"Pilot Symbol Length{len(pilot_symbol)} ")
    print(f"Pilot Ref len {len(pilot_ref_t)}")
    
    #Calc pilot cfo
    f_hat, final_f_idx, G_matrix, delay = om.cfo_calc(tx = pilot_ref_t, rx = pilot_symbol, FS = FreqConfig.FS, n_bins = FreqConfig.n_bins, delay_range=map.cp_len + 5)
    f_hat = FreqConfig.f_grid[final_f_idx]
    print(f"delay is {delay}")
    # for G in G_matrix:
    #     plt.figure()
    #     plt.plot(np.abs(G))
    #     plt.title("Abs G")
    #     plt.show()

  
    print(f"Pilot CFO Calc: {f_hat}")
    #Correct Pilot Symbol with Calced CFO
    #delay = 19
    pilot_symbol_corr = pilot_symbol[-(delay+map.N):-delay] * np.exp(-1j * 2*np.pi * f_hat * FreqConfig.n_pilot)

    #-------Channel Estimation--------
    pilot_recieved_k = np.fft.fft(pilot_symbol_corr)
    lambda_k = om.channel_estimation(recieved_pilot_symbol=pilot_recieved_k, known_pilot_symbol=pilot_ref_k)
    

    #------Correct for Channel Estiamtion-------
    pilot_final = pilot_recieved_k[SymbolConfig.data_idx] / lambda_k

    

    plt.figure()
    plt.plot(pilot_final, label="Final Pilot")
    plt.plot(pilot_ref_k[SymbolConfig.data_idx], label ="Ref")
    plt.legend()
    plt.title("Pilot Final")
    plt.show()

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(np.abs(lambda_k))
    plt.title("Magnitude")

    plt.subplot(2,1,2)
    plt.plot(np.angle(lambda_k))
    plt.title("Angle")
    plt.show()    

    qam_16_iq = qam_values()

    #Constalation Plot
    plt.subplot(4, 3, (8,15))
    plt.plot(np.real(pilot_final) * np.sqrt(10)  , np.imag(pilot_final) * np.sqrt(10), '.', label = "Recieved OFDM packet")
    plt.plot(np.real(qam_16_iq) * np.sqrt(10), np.imag(qam_16_iq) * np.sqrt(10), '.', label = "Constalation Map")
    plt.title("Constalation Diagram (16-QAM)")
    plt.xlabel("Real Axis")
    plt.ylabel("Imaginary Axis")
    plt.show()

def qam_values():
    qam_16 = ["0000",
              "0001",
              "0010",
              "0011",
              "0100",
              "0101",
              "0110",
              "0111",
              "1000",
              "1001",
              "1010",
              "1011",
              "1100",
              "1101",
              "1110",
              "1111"]
    
    qam_16_iq = []
    for word in qam_16:
        qam_16_iq.append(om.binary_to_iq(word))
    return qam_16_iq

if __name__ == "__main__":
    main()