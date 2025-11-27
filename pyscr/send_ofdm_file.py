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
def main():
    map = SubcarrierMap()

    # CLI ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', type=bool, default = False)
    parser.add_argument('--channel', '-c', type=str, default = "B")
    args = parser.parse_args()
    SIMULATION = args.sim
    CHANNEL_SEL = args.channel



    #Instantate OFDM Objects
    init_globals()
    #OFDM Packet File
    TX_FILE_PATH = "data_files/rand_ofdm_packet.dat"
    #Store read out file 
    RX_FILE_PATH = "data_files/rand_ofdm_packet_rx.dat"
    REF_FILE_PATH = "data_files/rand_ofdm_packet_ref.json"

    """
    FIXME: TAKING # OF DATA SYMBOLS FROM REFERSE .json file NEED TO IMPLEMENT THIS IS TRAINIG SYMBOL
    """
    with open(REF_FILE_PATH, 'r') as file:
        data = json.load(file)
        N_data_symbols = data["N_data_symbols"]
        n_samples = data["n_samples"]
        i_samples = data["i_data"]
        q_samples = data["q_data"]
    
    iq_ref_samples = np.array(i_samples, dtype=complex) + 1j* np.array(q_samples, dtype=complex)
    print(N_data_symbols)
    print(n_samples)
    #N_data_symbols = N_data_symbols + 1
    N_symbols = N_data_symbols + 1
    N_PILOT_SYMBOLS = 1
    N_SYNC_SYMBOLS = 1
    pilot_values = np.array([1,1,1,1], dtype=complex)
    pilots_idx = np.array([map.idx(k) for k in map.pilots_k])
    


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
    FILE_NAME = RX_FILE_PATH
    NSAMPLES = str(n_samples + SAMPLE_BUFFER)
    SETTLING = "0"
    TX_FILE = TX_FILE_PATH
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


    #----------
    #COARSE SYNQ
    #------------

    print("Unpacking OFDM Symbol...\n \n \n")
    file_size = os.path.getsize(file_name)
    iq = np.fromfile(file_name, dtype = np.complex64)
    print("Calculating M Values... \n")
    M_Values, P_Values = calc_M_values(iq)
    print("Done!\n")

    #Filter M Values
    M_filtered = filter_M(M_Values)
    sync_idx = np.argmax(M_filtered)
    start_idx = np.array([sync_idx]) + 1
    #Get starting point

    #=----------
    #Fine Synq
    #-----------

    #Generate Ref sync symbol
    known_sync_symbol = om.create_tx_block(SyncSymbol())
    refined_start_idx = fine_time_sync(iq, int(start_idx[0]), known_sync_symbol, search_window=30) + map.cp_len
    print(f"Coarse Index: {start_idx[0]} --> Refined Index: {refined_start_idx}")
    start_idx = np.array([refined_start_idx])

    zero_crossings = find_sync_start(M_Values, M_filtered)
    #start_idx = np.nonzero(zero_crossings)[0] + 2
    #Get Sync and Payload Idx
    #b_preamble = np.zeros(map.N + N_symbols * (map.N + map.cp_len))
    #b_preamble[:map.N] = 1
    
    #Estimate payload and preamble valid
    preamble_valid_est = estimate_preamble_valid(N_symbols, zero_crossings)
    payload_valid_est = estimate_payload_valid(N_symbols, N_data_symbols + 1, zero_crossings)
    
    #---------------
    # COARSE CFO CORRECTION
    #---------------
    #print(P_Values)
    print(f"START IDX = {start_idx}")
    current_start_idx = start_idx[0]
    P_peak = P_Values[current_start_idx]
    theta = np.angle(P_peak)
    f_hat = (theta * FS) / (np.pi * map.N)
    print(f"Estimated CFO: {f_hat} Hz")

    t_vector = np.arange(len(iq)) / FS

    correction_vector = np.exp(-1j * 2 * np.pi * f_hat * t_vector)
    #Apply correction
    iq = iq * correction_vector
    print("Coarse CFO Correction Applied")




    #Get the OFDM symbols - CP
    print(f"len iq = {len(iq)}")
    print(f" len payload_valid_est = {len(payload_valid_est)}")
    ofdm_symbols = get_ofdm_symbols(iq, payload_valid_est)
    #print(f"len of OFDM SYMBOLS IS: {len(ofdm_symbols)}")
    chunks = np.split(ofdm_symbols, N_data_symbols + 1)

    #Unpack total OFDM packet after starting point
    packet_len = (map.N + map.cp_len) * (N_data_symbols + N_PILOT_SYMBOLS + N_SYNC_SYMBOLS)
    ofdm_packet = iq[start_idx[0] : start_idx[0] + packet_len]
    symbols = np.split(ofdm_packet, N_data_symbols + N_PILOT_SYMBOLS + N_SYNC_SYMBOLS)

    symbols_no_cp = []
    for symbol in symbols:
        symbols_no_cp.append(symbol[:-map.cp_len])

    chunks = symbols_no_cp[N_SYNC_SYMBOLS + N_SYNC_SYMBOLS :]
    print(f"LENGTH OF CHUNKS = {len(chunks)}")
    #FIXME: BELOW ARE HARDCODED VALUES NEEDS TO BE UPDATED LATER
    sync_symbol = symbols_no_cp[0]
    pilot_symbol = symbols_no_cp[1]

    pilot_symbol_ref = PilotSymbol().symbol
    pilot_symbol_ref = np.fft.ifft(pilot_symbol_ref)

    G, f_hat = om.cfo_correct(Tx = pilot_symbol_ref, Rx = pilot_symbol, fs=FS)

    plt.figure()
    plt.plot(np.abs(G))
    plt.show()
    # pilot_symb_ref = om.create_tx_block(PilotSymbol())
    # pilot_symb_ref = pilot_symb_ref[map.cp_len:]
    # pilot_recieved = chunks[0]
    n_bins = 2 ** 14
    # f_grid = np.linspace(-FS/2 , FS/2 , n_bins)
    # G,f_hat = om.cfo_correct(Tx = pilot_symb_ref, Rx = (pilot_symbol), fs = FS, n_bins = n_bins)
    # print(f"FHAT IS {f_hat}")
    # plt.plot(f_grid, G)
    # plt.show()
    # #Do the phase change
    # n_len = len(iq)
    # n = np.arange(n_len)


    #----------------------
    #CHANNEL ESTIMATION
    #--------------------
    data_k = map.data_bins
    data_idx = np.array([map.idx(k) for k in data_k])
    pilot_symb_ref = PilotSymbol().symbol
    pilot_recieved = np.fft.fft(pilot_symbol)
    lambda_k = channel_estimation(recieved_pilot_symbol=pilot_recieved, known_pilot_symbol=pilot_symb_ref)


    #Build Reference Data Pilot Symbols
    TXk_pilot = dg.generate_random_packet(1)
    TXk_pilot = TXk_pilot[-map.N:]
    TXk_pilot = np.fft.fft(TXk_pilot)
    for i in range(map.N):
        if i not in pilots_idx:
            TXk_pilot[i] = 0 + 0j

    print(TXk_pilot)
    Y = []
    nt = np.arange(map.N)
    for chunk in chunks:

        #CFO Adjustment

        #Convert to Frequency Domain
        RXk = np.fft.fft(chunk)
        RXk_pilot = RXk.copy()
        for i in range(map.N):
            if i not in pilots_idx:
                RXk_pilot[i] = 0 + 0j

        txn_time = np.fft.ifft(TXk_pilot)
        rxn_time = np.fft.ifft(RXk_pilot)

        #Calculate G
        G, f_hat = om.cfo_correct(Tx = txn_time, Rx = rxn_time, fs=FS, n_bins = n_bins)
        print(f"FHAT is {f_hat}")
        # plt.figure()
        # plt.plot(np.abs(G))
        # plt.show()

        #Chanenl Estimation
        corrected_rx = chunk * np.exp(-1j * 2*np.pi * f_hat * nt / FS)

        chunk_fft = np.fft.fft(corrected_rx)
        #FIXME: FIX BELOW
        Y_tst = chunk_fft[data_idx]
        #Y_tst = corrected_RXk[data_idx]
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

    
    #CALCUALTE BIT ERROR RATE
    #bit_error_rate = om.calc_BER(ref_data, rx_binary)
    
    #CALCUALTE SYMBOL ERROR RATE (SER)
    #ser = om.calc_SER(ref_iq_16qam, Y_scaled)
    
    #Calculate ERROR VECTOR MAGNITUDE (EVM)
    #evm = om.calc_EVM(Y_scaled, ref_iq_16qam)



    #------------------------------
    # PLOT RESULTS
    #-----------------------------
    print("Plotting...")
    #Raw recieved time series data
    plt.figure()
    plt.subplot(4, 3, 1)
    plt.plot(iq)
    plt.title("Raw Recieved Data")
    plt.xlabel("Samples")
    
    #Plot of Sync Algorithm 
    plt.subplot(4, 3, 2)
    plt.plot(np.angle(lambda_k))
    plt.title("Lambda K Angle")
    plt.xlabel("Data bins")
#     plt.plot(iq, label = "OFDM Packet")
#     plt.plot(M_Values, label = "M Values")
#     plt.plot(M_filtered, label = "Filtered M")
#     #plt.plot(D, label = "Derivative of M_filter")
#     #plt.plot(zeroCrossing_3, label = "zero crossings")
#     #plt.plot(ignore_times, label = "Ignore window")
#    # plt.plot(actual_synq, label = "Actual Sync packet")
#     plt.plot(preamble_valid_est, label = "Estiamtion of valid Sync")
#     plt.plot(payload_valid_est, label = "Estimation of valid payload")
#     #plt.legend()
#     plt.title("Shmidil Cox Sync Algoirthm")
#     plt.xlabel("Samples")

    #Chanel Estimation lambda plot
    plt.subplot(4, 3 , 3)
    plt.plot(np.abs(np.fft.fftshift(lambda_k)))
    plt.title("Channel Estimation Gain Magnitude (Lambda)")
    plt.xlabel("Symbol Bins (k)")

    qam_16_iq = qam_values()

    #Constalation Plot
    plt.subplot(4, 3, (8,15))
    plt.plot(np.real(Y) * np.sqrt(10)  , np.imag(Y) * np.sqrt(10), '.', label = "Recieved OFDM packet")
    plt.plot(np.real(qam_16_iq) * np.sqrt(10), np.imag(qam_16_iq) * np.sqrt(10), '.', label = "Constalation Map")
    plt.title("Constalation Diagram (16-QAM)")
    plt.xlabel("Real Axis")
    plt.ylabel("Imaginary Axis")

    #----------------------
    #PRINT METRICS
    #-----------------------
    print(f"------- Metrics --------")
    # print(f"BER: {bit_error_rate}")
    # print(f"SER: {ser}")
    # print(f"EVM: {evm}")
    print((f"------------------------"))


    #Show plot
    plt.show()
    
    print("Plot Complete!")
    




#------------------
#Functions
#-----------------

def calc_M_values(iq_samples:np.ndarray):
    M_values = []
    P_values = []
    for i in range(len(iq_samples)):
        P, R, M = om.schmidl_cox_metrics_P_R_M(iq_samples, delay = i)
        M_values.append(M)
        P_values.append(P)
    return M_values, P_values


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

    if len(iq) != len(payload_valid_est):
        min_len = min(len(iq), len(payload_valid_est))
        iq = iq[:min_len]
        payload_valid_est = payload_valid_est[:min_len]
        print("############### get_ofdm_symbols VECTOR LENGTH MISMATCH ###################")
        print(f"Length of recieved iq samples {len(iq)} does not match length of calculated payload valid estimate {len(payload_valid_est)}")
        print(f"Truncating to both length = {min_len}")


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

def fine_time_sync(rx_signal, coarse_index, known_preamble, search_window = 20):

    N = len(known_preamble)

    #Define a search range around coarse index
    start_search = max(0, coarse_index - search_window)
    end_search = min(len(rx_signal), coarse_index + N + search_window)

    #Extract the chunk of received signal
    rx_chunk = rx_signal[start_search : end_search]

    #Cross correlate
    correlation = np.correlate(rx_chunk, known_preamble, mode='valid')

    #Find the peak cross correlation
    peak_offset = np.argmax(np.abs(correlation))

    #Calculate the true index
    true_index = start_search + peak_offset

    return true_index

if __name__ == "__main__":
    main()
