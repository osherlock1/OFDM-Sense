import subprocess
import numpy as np
import os
import matplotlib.pyplot as plt
from ofdm_manager import OFDMManager
from ofdm_symbol import OFDMSymbol
from subcarrier_map import SubcarrierMap
from data_generator import DataGenerator
from scipy.interpolate import interp1d
import json

# =========================
# Main
# =========================
def main():
    # Instantiate OFDM objects
    init_globals()

    # Paths
    tx_file_path = "data_files/rand_ofdm_packet.dat"
    rx_file_path = "data_files/rand_ofdm_packet_rx.dat"
    ref_file_path = "data_files/rand_ofdm_packet_ref.json"

    # Frame config (must match your generator)
    N_data_symbols = 5
    N = map.N
    Ncp = map.cp_len
    pad = 30  # generator adds 30 zeros head/tail

    # Expected TX length (time domain)
    expected_len = pad + (2*N + Ncp) + N_data_symbols*(N + Ncp) + pad
    NSAMPLES = str(int(1.5 * expected_len))  # margin

    # Pilots
    pilot_values = np.array([1, 1, 1, 1], dtype=complex)
    pilots_k = [-21, -7, 7, 21]  # centered indices

    # UHD args
    BUILD_PATH = "./build/TXRX_FROM_FILE"
    DEV_ADDR = "addr=192.168.30.2"
    TX_RATE = "1e6"
    RX_RATE = "1e6"
    TX_FREQ = "0"
    RX_FREQ = "0"
    TX_GAIN = "0"
    RX_GAIN = "0"
    OTW = "sc16"
    TYPE = "float"
    SETTLING = "0.1"
    TX_FILE = tx_file_path
    TX_TYPE = "float"
    TX_SPB = "4096"
    TX_REPEAT = "false"

    run_cmd = [
        BUILD_PATH,
        "--tx-args", DEV_ADDR,
        "--rx-args", DEV_ADDR,
        "--tx-rate", TX_RATE,
        "--rx-rate", RX_RATE,
        "--tx-freq", TX_FREQ,
        "--rx-freq", RX_FREQ,
        "--tx-gain", TX_GAIN,
        "--rx-gain", RX_GAIN,
        "--otw", OTW,
        "--type", TYPE,
        "--file", rx_file_path,
        "--nsamps", NSAMPLES,
        "--settling", SETTLING,
        "--tx-file", TX_FILE,
        "--tx-type", TX_TYPE,
        "--tx-spb", TX_SPB,
        "--tx-repeat", TX_REPEAT,
    ]

    print("\nRunning", BUILD_PATH)
    print(" ".join(run_cmd))
    subprocess.run(run_cmd, check=True)
    print(f"Run of {BUILD_PATH} complete.")

    # -------------------------
    # Read RX
    # -------------------------
    iq = np.fromfile(rx_file_path, dtype=np.complex64)
    if iq.size == 0:
        raise RuntimeError("RX file empty")

    # -------------------------
    # Sync + CFO (time domain)
    # -------------------------
    m0, _M = schmidl_cox_coarse_start(iq, N, Ncp)
    eps = schmidl_cox_cfo(iq, m0, N)          # fractional CFO (in subcarrier spacings)
    iq_c = cfo_correct(iq, eps)
    u = cp_refine(iq_c, m0, N, Ncp, U=16)      # small integer refine
    m = m0 + u

    chunks = slice_payload_symbols(iq_c, m, N, Ncp, N_data_symbols)
    if len(chunks) < N_data_symbols:
        raise RuntimeError("Not enough payload symbols after slicing")

    # -------------------------
    # Pilot-based equalization
    # -------------------------
    Y_eq_all = []
    for sym_td in chunks:
        Y = np.fft.fft(sym_td, N)  # unshifted FFT bins 0..N-1
        Xhat, delta, phi0, cpe = equalize_with_pilots(
            Y, pilots_k, pilot_values, map.data_bins, N
        )
        Y_eq_all.append(Xhat)

    Y_eq_all = np.concatenate(Y_eq_all)

    # -------------------------
    # Normalize constellation (16-QAM Es=10 ⇒ RMS=√10)
    # -------------------------
    target_rms = np.sqrt(10.0)
    rms = np.sqrt(np.mean(np.abs(Y_eq_all) ** 2)) + 1e-12
    Y_norm = Y_eq_all * (target_rms / rms)

    # -------------------------
    # Decode and metrics
    # -------------------------
    decoded_sym = om.decode_rx(Y_norm)

    ref_bits = unpack_json_ref(ref_file_path)
    rx_bits_str = ''.join(om.iq_to_binary(s, scale_factor=1) for s in decoded_sym)
    rx_bits = np.fromiter((1 if ch == '1' else 0 for ch in rx_bits_str), dtype=np.uint8)

    ber = om.calc_BER(ref_bits, rx_bits)
    print(f"BER: {ber}")

    parsed = dg._parse_string(rx_bits_str, 4)
    ref_iq = np.array([om.binary_to_iq(b) for b in parsed]) * target_rms
    ser = om.calc_SER(ref_iq, decoded_sym)
    print(f"SER: {ser}")

    # -------------------------
    # Plots (optional)
    # -------------------------
    qam_16_iq = np.array(qam_values()) * target_rms

    plt.figure()
    plt.plot(np.real(Y_norm), np.imag(Y_norm), '.', label="Received")
    plt.plot(np.real(qam_16_iq), np.imag(qam_16_iq), '.', label="16QAM")
    plt.axis('equal'); plt.legend(); plt.title("Constellation")

    plt.figure()
    tview = min(4*(N+Ncp), len(iq_c))
    plt.plot(iq_c.real[:tview], label="I")
    plt.plot(iq_c.imag[:tview], label="Q")
    plt.legend(); plt.title("Time-domain (CFO-corrected)")
    plt.show()

# =========================
# Helpers: sync + slicing
# =========================
def schmidl_cox_coarse_start(r, N, Ncp):
    """
    Coarse timing using a preamble with two identical halves of length N.
    Returns m0 at the start of the duplicated section.
    """
    L = N
    max_len = min(len(r), 4*(N+Ncp) + 2*L)
    if max_len < 2*L + 1:
        return 0, np.array([0.0])

    a0 = r[:max_len]           # 0..max_len-1
    a1 = r[L:max_len]          # L..max_len-1
    K = len(a1)                # = max_len - L

    # Element-wise product for offset-L correlation, then slide-sum length L
    p = a0[:K] * np.conj(a1)   # length K
    win = np.ones(L, dtype=float)
    if K < L:
        return 0, np.array([0.0])

    P = np.convolve(p, win, mode='valid')                # length K-L+1
    R = np.convolve(np.abs(a1)**2, win, mode='valid')    # same length

    M = (np.abs(P)**2) / (R + 1e-12)
    m0 = int(np.argmax(M))  # index in original stream
    return m0, M

def schmidl_cox_cfo(r, m0, N):
    """
    Fractional CFO estimate in subcarrier spacings from preamble halves.
    """
    L = N
    if m0 + 2*L > len(r):
        return 0.0
    a = np.vdot(r[m0:m0+L], r[m0+L:m0+2*L])
    return np.angle(a) / (2*np.pi)

def cfo_correct(r, eps):
    """
    Apply fractional CFO correction in time domain.
    eps: normalized by subcarrier spacing.
    """
    n = np.arange(len(r))
    return r * np.exp(-1j * 2*np.pi * eps * n)

def cp_refine(r, m0, N, Ncp, U=16):
    """
    Integer-sample refinement around m0 by maximizing CP correlation.
    """
    best_u, best_val = 0, -1.0
    for u in range(-U, U+1):
        a = r[m0+u : m0+u+Ncp]
        b = r[m0+u+N : m0+u+N+Ncp]
        if len(a) != Ncp or len(b) != Ncp:
            continue
        val = np.abs(np.vdot(a, b))
        if val > best_val:
            best_val, best_u = val, u
    return int(best_u)

def slice_payload_symbols(r, m, N, Ncp, N_data_symbols):
    """
    Deterministic slicing: [preamble(2N+CP)] then data symbols with CP.
    Returns list of N-length arrays (CP removed).
    """
    m_payload0 = m + 2*N + Ncp
    out = []
    for s in range(N_data_symbols):
        p = m_payload0 + s*(N+Ncp)
        sym = r[p+Ncp : p+Ncp+N]
        if len(sym) != N:
            break
        out.append(sym)
    return out

# =========================
# Helpers: EQ + pilots
# =========================
def k2i(k, N):
    return (np.asarray(k) % N).astype(int)

def estimate_slope_intercept_from_pilots(Y, pilots_k, pilots_sym, N):
    ip = k2i(pilots_k, N)
    Hp = Y[ip] / pilots_sym
    phi = np.unwrap(np.angle(Hp))
    a, b = np.polyfit(pilots_k, phi, 1)     # phi ≈ a*k + b
    delta_hat = -a * N / (2 * np.pi)        # timing offset in samples
    phi0_hat = b                             # CPE
    return delta_hat, phi0_hat

def derotate_delta_phi(Y, delta, phi0):
    N = Y.size
    k = np.arange(N)                         # unshifted bin indices
    deramp = np.exp(1j * (2 * np.pi * k * delta / N + phi0))
    return Y * deramp

def equalize_with_pilots(Y, pilots_k, pilots_sym, data_k, N):
    # Remove timing slope + CPE
    delta, phi0 = estimate_slope_intercept_from_pilots(Y, pilots_k, pilots_sym, N)
    Yc = derotate_delta_phi(Y, delta, phi0)

    # LS channel on pilots
    ip = k2i(pilots_k, N)
    Hp = Yc[ip] / pilots_sym

    # Interpolate channel to data carriers
    fH = interp1d(pilots_k, Hp, kind='linear', fill_value="extrapolate", assume_sorted=False)
    Hd = fH(np.asarray(data_k))

    # Equalize
    idata = k2i(data_k, N)
    Xhat = Yc[idata] / Hd

    # Residual CPE cleanup using pilots
    cpe = np.angle(np.mean((Yc[ip] / (Hp * pilots_sym))))
    Xhat *= np.exp(-1j * cpe)

    return Xhat, delta, phi0, cpe

# =========================
# Misc
# =========================
def init_globals():
    global map, om, dg
    map = SubcarrierMap()
    om = OFDMManager()
    dg = DataGenerator()

def qam_values():
    qam_16 = [
        "0000","0001","0010","0011",
        "0100","0101","0110","0111",
        "1000","1001","1010","1011",
        "1100","1101","1110","1111"
    ]
    return [om.binary_to_iq(word) for word in qam_16]

def unpack_json_ref(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    bits_str = ''.join(data['binary_data:'])
    return np.fromiter((1 if ch == '1' else 0 for ch in bits_str), dtype=np.uint8)

if __name__ == "__main__":
    main()
