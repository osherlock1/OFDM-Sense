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
def main():


    # CLI ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', type=bool, default = False)
    args = parser.parse_args()
    SIMULATION = args.sim



    #Instantate OFDM Objects
    init_globals()
    #OFDM Packet File
    TX_FILE_PATH = "data_files/rand_ofdm_packet.dat"
    #Store read out file 
    RX_FILE_PATH = "data_files/rand_ofdm_packet_rx.dat"
    REF_FILE_PATH = "data_files/rand_ofdm_packet_ref.json"
    """
    FIXME: HARDCODED DATA BELOW
    """
    N_data_symbols = 5
    N_symbols = N_data_symbols + 1
    pilot_values = np.array([1,1,1,1], dtype=complex)
    pilot_idx = [map.idx(map.pilots_k[i]) for i in range(len(pilot_values))]
    

    #CPP ARGS
    BUILD_PATH = "./build/TXRX_FROM_FILE"
    TX_ADDR = "addr=192.168.30.2"
    RX_ADDR = "addr=192.168.30.2"
    TX_RATE = "10e6"
    RX_RATE = "10e6"
    TX_FREQ = "8e6"
    RX_FREQ = "8e6"
    WAVE_TYPE = "SINE"
    WAVE_FREQ = "100e3"
    AMPL = "0.3"
    TX_GAIN = "0"
    RX_GAIN = "0"
    OTW = "sc16"
    TYPE = "float"
    FILE_NAME = RX_FILE_PATH
    NSAMPLES = "1000"
    SETTLING = "0"
    #TX_FILE = "data_files/usrp_samples_fc32_test.dat"
    TX_FILE = TX_FILE_PATH
    TX_TYPE = "float"
    TX_SPB = "0"
    TX_REPEAT = "false"


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
        "--tx-repeat", TX_REPEAT
    ]
    #-------------------------
    # TRANSFER FILE DATA OVER USRP
    #----------------------------

    if SIMULATION is False:
        print("\n")
        print(f"Running {BUILD_PATH}...")
        print(str(run_cmd))
        subprocess.run(run_cmd)
        print(f"Run of {BUILD_PATH} complete!")
        file_name = RX_FILE_PATH
    else:
        print("Simulating OFDM Transfer... \n")
        file_name = TX_FILE_PATH


    # ---------------------------------
    # UNPACK RX OFDM SYMBOL
    # --------------------------------

    print("Unpacking OFDM Symbol...\n \n \n")
    file_size = os.path.getsize(file_name)
    iq = np.fromfile(file_name, dtype = np.complex64)
    plt.figure()
    plt.plot(iq)
    print("Calculating M Values... \n")
    M_Values = calc_M_values(iq)
    print("Done!\n")

    #Filter M Values
    M_filtered = filter_M(M_Values)

    #Get starting point
    zero_crossings = find_sync_start(M_Values, M_filtered)

    #Get Sync and Payload Idx
    b_preamble = np.zeros(map.N + N_symbols * (map.N + map.cp_len))
    b_preamble[:map.N] = 1

    #Estimate payload and preamble valid
    preamble_valid_est = estimate_preamble_valid(N_symbols, zero_crossings)
    payload_valid_est = estimate_payload_valid(N_symbols, N_data_symbols, zero_crossings)
    
    #Get the OFDM symbols - CP
    ofdm_symbols = get_ofdm_symbols(iq, payload_valid_est)
    #print(f"len of OFDM SYMBOLS IS: {len(ofdm_symbols)}")
    chunks = np.split(ofdm_symbols, N_data_symbols)
    #chunk = chunks[0]
    #Perform FFT

    #----------------------
    #CHANNEL ESTIMATION
    #--------------------
    data_k = map.data_bins
    data_idx = np.array([map.idx(k) for k in data_k])
    pilot_symb_ref = PilotSymbol().symbol
    pilot_recieved = np.fft.fft(chunks[0])
    lambda_k = channel_estimation(pilot_recieved, pilot_symb_ref)
    plt.figure()
    plt.plot(np.fft.fftshift(lambda_k))
    plt.title("lambda k")
    #plt.show()

    # Y_test = np.fft.fft(chunks[1])
    # Y_test = Y_test[data_idx]
    # Y_test = Y_test / lambda_k


    Y = []
    data_chunks = chunks[1:]
    for chunk in data_chunks:
        chunk_fft = np.fft.fft(chunk)
        Y_tst = chunk_fft[data_idx]
        Y_tst = Y_tst / lambda_k
        Y.append(Y_tst)
    Y = np.concatenate(Y)
    Y_scaled = Y * np.sqrt(10)


    #-------------------------
    # CALCULATE METRICS
    #------------------------

    #Get golden reference data from json reference
    ref_data = unpack_json_ref(REF_FILE_PATH)
    ref_data = ref_data[:-192]

    #Convert Recieved Data to binary
    rx_binary = []
    for iq_sample in Y_scaled:
        rx_binary.append(om.iq_to_binary(iq_sample))
    rx_string = ''.join(rx_binary)
    rx_binary = np.array([int(bit) for bit in rx_string])

    #Convert Reference data from binary to IQ samples
    ref_list = [bit for bit in ref_data]
    ref_string = ""
    for bit in ref_list: #Convert the list of ints to single bit string
        ref_string += str(bit)
    
    #Parse Bit string and convert to iq samples
    ref_string_parsed = dg._parse_string(ref_string, 4)

    ref_iq = []
    for bits in ref_string_parsed:
        ref_iq.append(om.binary_to_iq(bits))
    ref_iq = np.array([ref_iq]) #Convert list to np array

    ref_iq_16qam = ref_iq * np.sqrt(10) #Scale ref_iq to 16qam
    ref_iq_16qam = ref_iq_16qam[0]

    

    print(f"------- Metrics --------")
    #CALCUALTE BIT ERROR RATE
    bit_error_rate = om.calc_BER(ref_data, rx_binary)
    print(f"BER: {bit_error_rate}")
    #CALCUALTE SYMBOL ERROR RATE (SER)
    ser = om.calc_SER(ref_iq_16qam, Y_scaled)
    print(f"SER: {ser}")
    #Calculate ERROR VECTOR MAGNITUDE (EVM)
    evm = om.calc_EVM(Y_scaled, ref_iq_16qam)
    print(f"EVM: {evm}")
    print((f"------------------------"))


    #------------------------------
    # PLOT RESULTS
    #-----------------------------


    qam_16_iq = qam_values()

    plt.figure()
    plt.plot(np.real(Y) * np.sqrt(10)  , np.imag(Y) * np.sqrt(10), '.', label = "Recieved OFDM packet")
    plt.plot(np.real(qam_16_iq) * np.sqrt(10), np.imag(qam_16_iq) * np.sqrt(10), '.', label = "Constalation Map")
    plt.show()



    plt.figure()
    plt.plot(iq, label = "OFDM Packet")
    plt.plot(M_Values, label = "M Values")
    plt.plot(M_filtered, label = "Filtered M")
    #plt.plot(D, label = "Derivative of M_filter")
    #plt.plot(zeroCrossing_3, label = "zero crossings")
    #plt.plot(ignore_times, label = "Ignore window")
   # plt.plot(actual_synq, label = "Actual Sync packet")
    plt.plot(preamble_valid_est, label = "Estiamtion of valid Sync")
    plt.plot(payload_valid_est, label = "Estimation of valid payload")
    plt.legend()
    plt.show()


#------------------
#Functions
#-----------------

def calc_M_values(iq_samples:np.ndarray):
    M_values = []
    for i in range(len(iq_samples)):
        P, R, M = om.schmidl_cox_metrics_P_R_M(iq_samples, delay = i)
        M_values.append(M)
    return M_values


def init_globals():
    global map, om, dg
    map = SubcarrierMap()
    om = OFDMManager()
    dg = DataGenerator()

def filter_M(M_values):
    ofdm_CP = map.cp_len
    b_toPeak = np.ones(ofdm_CP) / ofdm_CP
    a = (1,)
    M_filter = scipy.signal.lfilter(b_toPeak, a, M_values)
    return M_filter

def find_sync_start(M_values, M_filter):
    M_values_np = np.array(M_values, dtype=complex)

    #Get derivative of filter
    D = np.diff(M_filter)

    zeroCrossing_2 = ((D[:-1] * D[1:]) <= 0) * (M_values_np[1:-1] > 0.5)

    b_ignore = np.ones(1+map.N)
    b_ignore[0] = 0
    ignore_times = (scipy.signal.lfilter(b_ignore, (1, ), zeroCrossing_2) > 0).astype(int)

    zeroCrossing_3 = zeroCrossing_2 * (ignore_times == 0)
    
    return zeroCrossing_3

def estimate_preamble_valid(N_symbols, zeroCrossing_3):
    
    b_preamble = np.zeros(map.N + N_symbols * (map.N + map.cp_len))
    b_preamble[:map.N] = 1
    preamble_valid_est = scipy.signal.lfilter(b_preamble, (1,), zeroCrossing_3)
    return preamble_valid_est

def estimate_payload_valid(N_symbols, N_data_symbols, zeroCrossing_3):

    b_payload = np.zeros(map.N + N_symbols*(map.N + map.cp_len))
    for s in range(N_data_symbols):
        b_payload[map.N + (s + 1)*map.cp_len + s * map.N + np.arange(map.N)] = 1  
    
    payload_valid_est = scipy.signal.lfilter(b_payload, (1,), zeroCrossing_3)
    return payload_valid_est


def get_ofdm_symbols(iq, payload_valid_est):

    payload_idx = np.where(payload_valid_est >0)[0]

    rx_ofdm_symbols = []
    for idx in payload_idx:
        rx_ofdm_symbols.append(iq[idx])
    return np.array(rx_ofdm_symbols, dtype=complex)



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

def centered_bins(N):
    k = np.arange(N)
    k[k >= N//2] -= N     # [-N/2..N/2-1]
    return k

def k2i(k, N):
    # map centered k to unshifted FFT index i in [0..N-1]
    return (k % N).astype(int)

def estimate_delta_from_pilots(Y, pilots_k, pilots_sym, N):
    """
    Y          : length-N FFT of one OFDM symbol (unshifted order)
    pilots_k   : pilot subcarrier indices in centered form (e.g., [-21,-7,7,21])
    pilots_sym : known pilot symbols at those indices
    N          : FFT size
    Returns: delta_hat in samples
    """
    i = k2i(np.asarray(pilots_k), N)
    Hp = Y[i] / pilots_sym                # observed channel on pilots (incl. timing ramp)
    phi = np.unwrap(np.angle(Hp))         # pilot phases
    a, b = np.polyfit(pilots_k, phi, 1)   # phase ~ a*k + b
    delta_hat = - a * N / (2*np.pi)
    return delta_hat

def adjust_offset(delta, Y):
    N = Y.size
    i = np.arange(N)                      # unshifted bin indices
    deramp = np.exp(1j * 2*np.pi * i * delta / N)
    return Y * deramp

def unpack_json_ref(file_name):

    with open(file_name, 'r') as file:
        data = json.load(file)
    
    binary_data = data['binary_data:']
    binary_data = np.array(binary_data)
    binary_data = ''.join(binary_data)

    binary_data = np.array([int(bit) for bit in binary_data])

    return binary_data

def channel_estimation(recieved_pilot_symbol:np.ndarray, known_pilot_symbol:np.ndarray):
    
    data_k = map.data_bins
    data_idx = np.array([map.idx(k) for k in data_k])
    
    
    r = recieved_pilot_symbol[data_idx]
    s = known_pilot_symbol[data_idx]
    s_conj = np.conj(s)
    sqr_mag_s = np.abs(s) ** 2
    channel_gain = (r * s_conj) / sqr_mag_s
    return channel_gain



if __name__ == "__main__":
    main()
